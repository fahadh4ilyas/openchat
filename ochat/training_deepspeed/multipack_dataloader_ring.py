from typing import Any, Optional, List, Callable

import torch.distributed as dist

import numpy as np
import numba


@numba.njit
def ffd_check(a: np.ndarray, c: int, n: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = np.full((n, ), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        if not_found:
            return False

    return True


@numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins = []
    bins_result = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])

    return bins_result


@numba.njit
def allocate(lengths: np.ndarray, lengths_cumsum: np.ndarray, c: int):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    s = 0
    start_index = 0
    rank = 0
    n = 1
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if ffd_check(lengths[start_index: start_index + m], c, n):
                l = m
            else:
                r = m

        # use length l
        batch = ffd_with_result(lengths[start_index: start_index + l], c, start_index)  # type: ignore
        if len(batch) < n:
            break

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])

    return result, s, len(result) * c * n


def extract_local(value, rank, world_size):
    value_chunks = np.split(value, 2 * world_size)
    local_value = np.concatenate(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]]
    )
    return local_value


class MultipackDistributedDataloader:
    """Unpadded data loading using Multipack.
       Approximate (at most ~1.22x) the optimal solution of the identical-machines scheduling problem, which is NP-hard."""
    
    def __init__(
        self,
        dataset: Any,
        lengths: np.ndarray,
        numseqs: np.ndarray,

        batch_max_length: int,
        collate_fn: Callable,

        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,

        seed: int = 0,
    ):
        # Dataset
        self.dataset = dataset
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

        self.batch_max_length = batch_max_length
        self.collate_fn = collate_fn

        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        # Seed
        self.seed = seed

        # Epoch
        self.epoch = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.default_rng(seed=self.seed + self.epoch).permutation(len(self.lengths))

        lengths        = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate(lengths=lengths,
                                                    lengths_cumsum=lengths_cumsum,
                                                    c=self.batch_max_length)
        
        batched_indices = [indices[batch]         for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        return batched_indices
    
    def prepare_dataset(self, dataset: dict):

        dataset['seqlens'] = dataset['seqlens'] // self.num_replicas

        keys = ["nz_input_ids", "nz_position_ids", "nz_shifted_label_ids", "nz_shifted_loss_weights"]

        for k in keys:
            for i in range(len(dataset[k])):
                dataset[k][i] = extract_local(dataset[k][i], self.rank, self.num_replicas)

        return dataset

    def __iter__(self):
        all_indices = self.generate_batches(set_stats=True)

        for indices in all_indices:
            dataset = self.dataset[indices]
            dataset = self.prepare_dataset(dataset)
            yield self.collate_fn(dataset)

    def num_batches(self):
        batches, _, _ = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
