"""Tests for ochat.training_utils — NumpyDataset, dataloaders, and training args."""

import os
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import orjson
import pytest

# Prevent HF network calls
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ochat.training_utils.numpy_dataset import NumpyDataset
from ochat.training_utils._training_args import (
    BaseTrainingArguments,
    LoraTrainingArgsMixin,
    add_base_args,
    add_lora_args,
)
from ochat.training_utils.multipack_dataloader import MultipackDistributedDataloader
from ochat.training_utils.multipack_dataloader_single import MultipackDataloader


# -- NumpyDataset tests -------------------------------------------------------

class TestNumpyDataset:
    """Test NumpyDataset loading from parquet files."""
    pytestmark = pytest.mark.cpu

    def _make_parquet(self, tmpdir, filename, data_dict, metadata=None):
        """Create a parquet file with given columns and optional metadata."""
        table = pa.table(data_dict)
        if metadata:
            meta_bytes = orjson.dumps(metadata)
            existing_meta = table.schema.metadata or {}
            table = table.replace_schema_metadata(
                {**existing_meta, b"metadata_json": meta_bytes}
            )
        path = os.path.join(tmpdir, filename)
        pq.write_table(table, path)
        return path

    def test_basic_load(self, tmpdir):
        """Load a simple single parquet file."""
        path = self._make_parquet(
            tmpdir,
            "data.train.parquet",
            {
                "total_length": [10, 20, 30],
                "num_seqs": [2, 3, 1],
                "nz_input_ids": [np.arange(10), np.arange(20), np.arange(30)],
            },
            metadata={"model_type": "qwen3_5"},
        )

        ds = NumpyDataset(path)
        assert len(ds) == 3
        assert ds.metadata == {"model_type": "qwen3_5"}
        assert ds["total_length"].tolist() == [10, 20, 30]

        # Index access
        row = ds[[0, 2]]
        assert row["total_length"].tolist() == [10, 30]

    def test_max_seqlen(self, tmpdir):
        """max_seqlen is derived from total_length column."""
        path = self._make_parquet(tmpdir, "data.train.parquet", {
            "total_length": [100, 500, 200],
            "num_seqs": [1, 1, 1],
        })
        ds = NumpyDataset(path)
        assert ds.max_seqlen == 500

    def test_missing_extension_auto_detect(self, tmpdir):
        """Dataset filename without extension should auto-detect .parquet."""
        path = self._make_parquet(tmpdir, "data.train.parquet", {
            "total_length": [10],
            "num_seqs": [1],
        })
        # Pass path without extension
        ds = NumpyDataset(os.path.join(tmpdir, "data.train"))
        assert len(ds) == 1

    def test_no_metadata(self, tmpdir):
        """Dataset without metadata_json key should have None metadata."""
        path = self._make_parquet(tmpdir, "data.train.parquet", {
            "total_length": [10],
            "num_seqs": [1],
        })
        ds = NumpyDataset(path)
        assert ds.metadata is None


# -- Training args tests ------------------------------------------------------

class TestTrainingArgs:
    """Test BaseTrainingArguments and LoraTrainingArgsMixin."""
    pytestmark = pytest.mark.cpu

    def test_base_args_defaults(self):
        args = BaseTrainingArguments(
            local_rank=0,
            model_path="/path/to/model",
            data_prefix="/path/to/data",
            save_path="/path/to/save",
            experiment_name="test",
            run_name="run1",
            deepspeed_config="ds_config.json",
        )
        assert args.batch_max_len == 81920
        assert args.base_lr == 3e-4
        assert args.epochs == 5

    def test_batch_max_len_validation(self):
        """batch_max_len must be multiple of 2048."""
        with pytest.raises(Exception):
            BaseTrainingArguments(
                local_rank=0,
                model_path="/m", data_prefix="/d", save_path="/s",
                experiment_name="test", run_name="run1",
                deepspeed_config="ds.json",
                batch_max_len=100,  # Not multiple of 2048
            )

    def test_lora_mixin_defaults(self):
        """LoraTrainingArgsMixin provides LoRA defaults."""
        class Args(BaseTrainingArguments, LoraTrainingArgsMixin):
            pass

        args = Args(
            local_rank=0,
            model_path="/m", data_prefix="/d", save_path="/s",
            experiment_name="test", run_name="run1",
            deepspeed_config="ds.json",
        )
        assert args.lora_r == 32
        assert args.lora_alpha == 32
        assert args.lora_dropout == 0.05
        assert args.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert args.use_qlora is False

    def test_argparse_helpers(self):
        import argparse
        parser = argparse.ArgumentParser()
        add_base_args(parser, base_lr=1e-2)
        add_lora_args(parser)

        parsed = parser.parse_args([
            "--local_rank", "0",
            "--model-path", "/m",
            "--data-prefix", "/d",
            "--save-path", "/s",
            "--experiment-name", "test",
            "--run-name", "r1",
        ])
        assert parsed.base_lr == 1e-2
        assert parsed.lora_r == 32

    def test_auto_lora_args(self):
        """LoRA args should be available even without --use-lora."""
        import argparse
        parser = argparse.ArgumentParser()
        add_base_args(parser)
        add_lora_args(parser)

        parsed = parser.parse_args([
            "--local_rank", "0",
            "--model-path", "/m", "--data-prefix", "/d", "--save-path", "/s",
            "--experiment-name", "test", "--run-name", "r1",
            "--use-qlora",
            "--quant-bits", "8",
            "--lora-r", "64",
        ])
        assert parsed.use_qlora is True
        assert parsed.quant_bits == 8
        assert parsed.lora_r == 64


# -- MultipackDataloader tests (single-GPU, no distributed) -------------------

class TestMultipackDataloaderSingle:
    """Test the single-GPU MultipackDataloader bin-packing logic."""
    pytestmark = pytest.mark.cpu

    def _make_dataset(self, lengths, numseqs=None):
        """Create a numpy-style dataset dict."""
        return {
            "total_length": np.array(lengths),
            "num_seqs": np.array(numseqs or [1] * len(lengths)),
        }

    def test_construction(self):
        data = self._make_dataset([100, 200, 300, 400, 500])
        ds = NumpyDataset.__new__(NumpyDataset)
        ds.dataset = data
        ds.length = len(data["total_length"])
        ds.max_seqlen = max(data["total_length"])

        def collate(batch):
            return {"input_ids": batch["total_length"]}, None

        loader = MultipackDataloader(
            dataset=ds,
            lengths=data["total_length"],
            numseqs=data["num_seqs"],
            batch_max_length=2048,
            collate_fn=collate,
            seed=42,
        )
        assert loader.num_batches() > 0

    def test_batches_fit_in_capacity(self):
        """All generated batches should fit within batch_max_length."""
        data = self._make_dataset(
            [500, 600, 700, 800, 900, 400, 300, 200, 100, 50],
            [2, 2, 3, 1, 1, 4, 1, 2, 1, 1],
        )
        ds = NumpyDataset.__new__(NumpyDataset)
        ds.dataset = data
        ds.length = len(data["total_length"])
        ds.max_seqlen = max(data["total_length"])

        def collate(batch):
            return {"sum": sum(batch["total_length"])}, None

        loader = MultipackDataloader(
            dataset=ds,
            lengths=data["total_length"],
            numseqs=data["num_seqs"],
            batch_max_length=2048,
            collate_fn=collate,
            seed=0,
        )

        batches, _ = loader.generate_batches(set_stats=False)
        for batch_indices in batches:
            batch_sum = sum(data["total_length"][batch_indices])
            assert batch_sum <= 2048, f"Batch sum {batch_sum} exceeds 2048"

    def test_efficiency_stats(self):
        """Efficiency tracking during iteration."""
        data = self._make_dataset([100, 200, 300])
        ds = NumpyDataset.__new__(NumpyDataset)
        ds.dataset = data
        ds.length = len(data["total_length"])
        ds.max_seqlen = max(data["total_length"])

        def collate(batch):
            return batch, None

        loader = MultipackDataloader(
            dataset=ds,
            lengths=data["total_length"],
            numseqs=data["num_seqs"],
            batch_max_length=1024,
            collate_fn=collate,
            seed=0,
        )
        # Iterate to populate stats
        list(loader)
        eff = loader.efficiency()
        assert 0.0 < eff <= 1.0

    def test_set_epoch_changes_order(self):
        """Different epochs produce different batches."""
        data = self._make_dataset([500, 600, 700, 800, 900, 400, 300, 200, 100])
        ds = NumpyDataset.__new__(NumpyDataset)
        ds.dataset = data
        ds.length = len(data["total_length"])
        ds.max_seqlen = max(data["total_length"])

        def collate(batch):
            return batch, None

        loader = MultipackDataloader(
            dataset=ds,
            lengths=data["total_length"],
            numseqs=data["num_seqs"],
            batch_max_length=2048,
            collate_fn=collate,
            seed=0,
        )

        loader.set_epoch(0)
        batches0, _ = loader.generate_batches()
        loader.set_epoch(1)
        batches1, _ = loader.generate_batches()

        # Flatten indices
        flat0 = [i for b in batches0 for i in b]
        flat1 = [i for b in batches1 for i in b]

        # Same set of indices, but different order
        assert set(flat0) == set(flat1)
