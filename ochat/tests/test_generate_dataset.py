"""Tests for ochat.data — generate_dataset, generate_dpo_dataset, pickle_dataset.

Unit tests for pure-Python utility functions and Pydantic argument validation.
Tokenization tests require GPU; data-shaping logic is tested without a model.
"""

import os
import tempfile
import json
import pytest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# -- generate_dataset DataArguments -------------------------------------------

class TestGenerateDatasetArgs:
    """Validate SFT generate_dataset DataArguments."""
    pytestmark = pytest.mark.cpu

    @pytest.fixture
    def base_kwargs(self):
        return dict(
            model_path="Qwen/Qwen3.5-0.8B",
            model_type="qwen3_5",
            in_files=["data.jsonl"],
            out_prefix="/tmp/data",
        )

    def test_valid_defaults(self, base_kwargs):
        from ochat.data.generate_dataset import DataArguments
        args = DataArguments(**base_kwargs)
        assert args.seed == 42
        assert args.eval_ratio == 0.0
        assert args.max_seq_length is None

    def test_max_jobs_validation(self, base_kwargs):
        """max_jobs must be <= max_workers when max_workers is set."""
        from ochat.data.generate_dataset import DataArguments
        with pytest.raises(Exception):
            DataArguments(**{**base_kwargs, "max_workers": 2, "max_jobs": 5})

    def test_max_jobs_ok_when_no_workers(self, base_kwargs):
        """max_jobs alone is fine (max_workers is None)."""
        from ochat.data.generate_dataset import DataArguments
        args = DataArguments(**{**base_kwargs, "max_jobs": 100})
        assert args.max_jobs == 100

    def test_per_sequence_loss_flag(self, base_kwargs):
        from ochat.data.generate_dataset import DataArguments
        args = DataArguments(**{**base_kwargs, "per_sequence_loss": True})
        assert args.per_sequence_loss is True


# -- generate_dpo_dataset DataArguments ---------------------------------------

class TestGenerateDPODatasetArgs:
    """Validate DPO generate_dpo_dataset DataArguments."""
    pytestmark = pytest.mark.cpu

    @pytest.fixture
    def base_kwargs(self):
        return dict(
            model_path="Qwen/Qwen3.5-0.8B",
            model_type="qwen3_5",
            in_files=["data_dpo.jsonl"],
            out_prefix="/tmp/dpo_data",
        )

    def test_no_ref_logps_flag(self, base_kwargs):
        from ochat.data.generate_dpo_dataset import DataArguments
        args = DataArguments(**{**base_kwargs, "no_ref_logps": True})
        assert args.no_ref_logps is True

    def test_valid_defaults(self, base_kwargs):
        from ochat.data.generate_dpo_dataset import DataArguments
        args = DataArguments(**base_kwargs)
        assert args.max_workers is None
        assert args.max_jobs == 10

    def test_dpo_pair_model(self):
        from ochat.data.generate_dpo_dataset import DPOPair
        pair = DPOPair(
            chosen={"items": [{"role": "user", "content": "Q"}]},
            rejected={"items": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Bad"}]},
        )
        assert pair.chosen["items"][0]["role"] == "user"
        assert len(pair.rejected["items"]) == 2


# -- truncate_trailing_zero_weighted ------------------------------------------

class TestTruncateTrailingZeros:
    """Test truncation of trailing zero-weighted tokens."""
    pytestmark = pytest.mark.cpu

    def _get_fn(self):
        from ochat.data.generate_dataset import truncate_trailing_zero_weighted
        return truncate_trailing_zero_weighted

    def test_no_trailing_zeros(self):
        tokens, weights = self._get_fn()([1, 2, 3], [0.0, 1.0, 0.5])
        assert tokens == [1, 2, 3]
        assert weights == [0.0, 1.0, 0.5]

    def test_trailing_zeros_removed(self):
        tokens, weights = self._get_fn()([1, 2, 3, 4], [1.0, 0.0, 0.0, 0.0])
        assert tokens == [1]
        assert weights == [1.0]

    def test_all_zeros(self):
        tokens, weights = self._get_fn()([1, 2, 3], [0.0, 0.0, 0.0])
        assert tokens == []
        assert weights == []

    def test_empty(self):
        tokens, weights = self._get_fn()([], [])
        assert tokens == []
        assert weights == []


# -- add_single_conv / _build_single_conv -------------------------------------

class TestSingleConvBuilder:
    """Test the data-shaping logic for a single conversation entry."""
    pytestmark = pytest.mark.cpu

    def _make_args(self, **overrides):
        from ochat.data.generate_dataset import DataArguments
        return DataArguments(
            model_path="Qwen/Qwen3.5-0.8B",
            model_type="qwen3_5",
            in_files=["data.jsonl"],
            out_prefix="/tmp/data",
            **overrides,
        )

    def test_basic_sft_build(self):
        """SFT add_single_conv: labels shifted, padded."""
        from ochat.data.generate_dataset import add_single_conv
        args = self._make_args(data_length_multiple_of=8, ignore_index=-100)

        outputs = []
        add_single_conv(
            outputs, tokens=[5, 10, 20, 30],
            weights=[0.0, 1.0, 1.0, 0.5],
            images=[], videos=[], args=args,
        )

        assert len(outputs) == 1
        out = outputs[0]

        # tokens should be padded to multiple of 8
        assert len(out["nz_input_ids"]) == 8
        # First 4 original tokens
        assert out["nz_input_ids"][:4] == [5, 10, 20, 30]
        # labels: shifted left, PAD at end
        assert out["nz_shifted_label_ids"][0] == 10   # token 10 with weight 1.0 → kept
        assert out["nz_shifted_label_ids"][1] == 20   # token 20 with weight 1.0 → kept
        assert out["nz_shifted_label_ids"][2] == 30   # token 30 with weight 0.5 → kept
        assert out["nz_shifted_label_ids"][3] == -100  # PAD
        # weights: shifted left, PAD weight at end
        assert out["nz_shifted_loss_weights"][0] == 1.0
        assert out["nz_shifted_loss_weights"][1] == 1.0
        assert out["nz_shifted_loss_weights"][2] == 0.5
        assert out["nz_shifted_loss_weights"][3] == 0.0

    def test_dpo_build(self):
        """DPO _build_single_conv: same logic, different output keys."""
        from ochat.data.generate_dpo_dataset import _build_single_conv
        args = self._make_args(data_length_multiple_of=4, ignore_index=-100)

        result = _build_single_conv(
            tokens=[1, 2, 3], weights=[0.0, 1.0, 1.0],
            images=[], videos=[], args=args,
        )
        assert result is not None
        assert len(result["nz_input_ids"]) == 4  # padded to 4
        assert result["nz_shifted_label_ids"][0] == 2
        assert result["nz_shifted_label_ids"][1] == 3
        assert result["nz_shifted_label_ids"][2] == -100  # PAD label

    def test_truncation_before_build(self):
        """Trailing zeros removed before build."""
        from ochat.data.generate_dataset import add_single_conv
        args = self._make_args(data_length_multiple_of=4, ignore_index=-100)

        outputs = []
        add_single_conv(
            outputs, tokens=[7, 8, 9, 10],
            weights=[0.0, 1.0, 0.0, 0.0],  # trailing zeros
            images=[], videos=[], args=args,
        )
        # Only [7, 8] kept after truncation
        out = outputs[0]
        assert out["nz_input_ids"][:2] == [7, 8]
        # Padded to 4
        assert len(out["nz_input_ids"]) == 4

    def test_ignore_last_token(self):
        """ignore_last_token mode: last token moved to padding."""
        from ochat.data.generate_dataset import add_single_conv
        args = self._make_args(
            data_length_multiple_of=8, ignore_index=-100, ignore_last_token=True
        )
        outputs = []
        add_single_conv(
            outputs, tokens=[1, 2, 3], weights=[0.0, 1.0, 1.0],
            images=[], videos=[], args=args,
        )
        out = outputs[0]
        # Length = 3, ignore_last_token → length becomes 2
        # Then padded to 8: [1, 2, 3(padding copy), 0...]
        assert len(out["nz_input_ids"]) == 8
        assert out["nz_input_ids"][0] == 1
        assert out["nz_input_ids"][1] == 2
        assert out["nz_input_ids"][2] == 3  # last token reused as padding anchor

    def test_single_token_sequence(self):
        """Edge case: single token with weight > 0."""
        from ochat.data.generate_dataset import add_single_conv
        args = self._make_args(data_length_multiple_of=4, ignore_index=-100)

        outputs = []
        add_single_conv(outputs, tokens=[42], weights=[1.0], images=[], videos=[], args=args)
        out = outputs[0]
        assert len(out["nz_input_ids"]) == 4
        assert out["nz_input_ids"][0] == 42
        # labels: shifted left, PAD
        assert out["nz_shifted_label_ids"][0] == -100  # PAD (shifted from single token)


# -- pickle_dataset ------------------------------------------------------------

class TestPickleDataset:
    """Test pickle_dataset.make_dict and NumpyDataset round-trip."""
    pytestmark = pytest.mark.cpu

    def _make_parquet(self, tmpdir, filename, data_dict, metadata=None):
        table = pa.table(data_dict)
        if metadata:
            import orjson
            meta_bytes = orjson.dumps(metadata)
            existing = table.schema.metadata or {}
            table = table.replace_schema_metadata({**existing, b"metadata_json": meta_bytes})
        path = os.path.join(tmpdir, filename)
        pq.write_table(table, path)
        return path

    def test_make_dict(self, tmpdir):
        """Convert parquet to dict format used by pickle."""
        from ochat.data.pickle_dataset import make_dict
        path = self._make_parquet(tmpdir, "data.train.parquet", {
            "total_length": [10, 20],
            "num_seqs": [1, 2],
        }, metadata={"model_type": "qwen3_5"})

        result = make_dict(path)
        assert result["length"] == 2
        assert result["metadata"] == {"model_type": "qwen3_5"}
        assert "total_length" in result["dataset"]
        assert result["dataset"]["total_length"].tolist() == [10, 20]

    def test_round_trip_via_numpy_dataset(self, tmpdir):
        """Parquet → dict → NumpyDataset should preserve data."""
        from ochat.data.pickle_dataset import make_dict
        from ochat.training_utils.numpy_dataset import NumpyDataset

        path = self._make_parquet(tmpdir, "data.train.parquet", {
            "total_length": [100, 200, 300],
            "num_seqs": [3, 1, 2],
        })

        # A pickle file would store this dict. NumpyDataset can read parquet
        # directly; we test that make_dict output matches direct NumpyDataset read.
        result = make_dict(path)
        ds = NumpyDataset(path)

        assert result["length"] == len(ds)
        for key in result["dataset"]:
            np.testing.assert_array_equal(result["dataset"][key], ds[key])

    def test_no_metadata(self, tmpdir):
        """Parquet without metadata → metadata is None."""
        from ochat.data.pickle_dataset import make_dict
        path = self._make_parquet(tmpdir, "data.train.parquet", {
            "total_length": [42],
            "num_seqs": [1],
        })
        result = make_dict(path)
        assert result["metadata"] is None


# -- generate_orpo_dataset (alias) --------------------------------------------

class TestGenerateOrpoDataset:
    """ORPO generate script: thin wrapper around DPO pipeline."""
    pytestmark = pytest.mark.cpu

    def test_imports_dpo(self):
        """generate_orpo_dataset reuses DPO's DataArguments and generate_dataset."""
        from ochat.data.generate_orpo_dataset import DataArguments, generate_dataset
        from ochat.data.generate_dpo_dataset import DataArguments as DPODataArgs
        from ochat.data.generate_dpo_dataset import generate_dataset as DPOGenerate
        assert DataArguments is DPODataArgs
        assert generate_dataset is DPOGenerate
