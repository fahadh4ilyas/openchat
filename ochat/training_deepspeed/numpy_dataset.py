import pyarrow.parquet as pq
import os
import numpy as np
import pickle
import orjson
from typing import Union, Iterable


class NumpyDataset:
    def __init__(self, dataset_filename: str):
        super().__init__()

        if dataset_filename.endswith(".pickle"):
            with open(dataset_filename, "rb") as f:
                data = pickle.load(f)
            self.dataset = data["dataset"]
            self.length = data["length"]
            self.metadata = data["metadata"]
        elif dataset_filename.endswith(".parquet"):
            # Convert parquet to numpy for fast random access
            table = pq.read_table(dataset_filename, memory_map=True)
            self.dataset = {
                k: v.to_numpy() for k, v in zip(table.column_names, table.columns)
            }
            self.length = table.num_rows

            # read metadata
            self.metadata = table.schema.metadata.get(b"metadata_json", None)
            if self.metadata is not None:
                self.metadata = orjson.loads(self.metadata)
        elif os.path.isfile(f"{dataset_filename}.pickle"):
            with open(f"{dataset_filename}.pickle", "rb") as f:
                data = pickle.load(f)
            self.dataset = data["dataset"]
            self.length = data["length"]
            self.metadata = data["metadata"]
        elif os.path.isfile(f"{dataset_filename}.parquet"):
            # Convert parquet to numpy for fast random access
            table = pq.read_table(f"{dataset_filename}.parquet", memory_map=True)
            self.dataset = {
                k: v.to_numpy() for k, v in zip(table.column_names, table.columns)
            }
            self.length = table.num_rows

            # read metadata
            self.metadata = table.schema.metadata.get(b"metadata_json", None)
            if self.metadata is not None:
                self.metadata = orjson.loads(self.metadata)
        elif os.path.isfile(f"{dataset_filename}.part000.pickle"):
            index = 0
            datasets = []
            lengths = []
            metadata = None
            while True:
                part_filename = f"{dataset_filename}.part{index:03d}.pickle"
                if not os.path.isfile(part_filename):
                    break
                with open(part_filename, "rb") as f:
                    data = pickle.load(f)
                datasets.append(data["dataset"])
                lengths.append(data["length"])
                if metadata is None:
                    metadata = data["metadata"]
                index += 1
            # concatenate datasets
            self.dataset = {}
            for k in datasets[0].keys():
                self.dataset[k] = np.concatenate([d[k] for d in datasets], axis=0)
            self.length = sum(lengths)
            self.metadata = metadata
        elif os.path.isfile(f"{dataset_filename}.part000.parquet"):
            index = 0
            datasets = []
            while True:
                part_filename = f"{dataset_filename}.part{index:03d}.parquet"
                if not os.path.isfile(part_filename):
                    break
                table = pq.read_table(part_filename, memory_map=True)
                dataset_part = {
                    k: v.to_numpy() for k, v in zip(table.column_names, table.columns)
                }
                datasets.append(dataset_part)
                index += 1
            # concatenate datasets
            self.dataset = {}
            for k in datasets[0].keys():
                self.dataset[k] = np.concatenate([d[k] for d in datasets], axis=0)
            self.length = sum([table.num_rows for table in datasets])
            # read metadata from the last part
            self.metadata = table.schema.metadata.get(b"metadata_json", None)
            if self.metadata is not None:
                self.metadata = orjson.loads(self.metadata)
        else:
            raise FileNotFoundError('Can not open the file!')

        self.max_seqlen = max(self.dataset["total_length"])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, indices: Union[str, Iterable[int]]):
        if isinstance(indices, str):
            return self.dataset[indices]

        return {k: v[indices] for k, v in self.dataset.items()}
