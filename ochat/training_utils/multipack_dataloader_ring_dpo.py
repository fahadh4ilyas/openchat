"""DPO ring-attention multipack dataloader.

Extends the training_utils ring dataloader to handle chosen_* / rejected_* prefixed
keys, splitting both sides of each DPO pair across ring-attention ranks.
"""

from typing import Any, Optional, Callable, Dict, Tuple
from numbers import Number

import torch
import torch.distributed as dist

import numpy as np

from ochat.training_utils.multipack_dataloader_ring import (
    MultipackDistributedDataloader as _RingDataloader,
    extract_local,
)


class MultipackDistributedDataloader(_RingDataloader):
    """Ring-attention dataloader for DPO paired data.

    Same bin-packing as the standard ring dataloader, but prepare_dataset also
    splits both chosen_* and rejected_* key families across ring-attention ranks.
    """

    _RING_KEYS = [
        "nz_input_ids",
        "nz_position_ids",
        "nz_shifted_label_ids",
        "nz_shifted_loss_weights",
    ]

    def __init__(
        self,
        dataset: Any,
        lengths: np.ndarray,
        batch_max_length: int,
        collate_fn: Callable[
            ..., Tuple
        ],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__(
            dataset=dataset,
            lengths=lengths,
            batch_max_length=batch_max_length,
            collate_fn=collate_fn,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )

    def prepare_dataset(self, dataset: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Split both chosen and rejected sequences across ring-attention ranks."""
        for prefix in ("chosen_", "rejected_"):
            seq_key = f"{prefix}seqlens"
            if seq_key in dataset:
                dataset[seq_key] = dataset[seq_key] // self.num_replicas
            for k in self._RING_KEYS:
                key = f"{prefix}{k}"
                if key in dataset:
                    for i in range(len(dataset[key])):
                        dataset[key][i] = extract_local(
                            dataset[key][i], self.rank, self.num_replicas
                        )
        return dataset
