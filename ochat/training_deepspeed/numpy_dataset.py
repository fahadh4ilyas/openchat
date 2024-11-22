import pyarrow.parquet as pq
import os
import pickle
import orjson
from typing import Union, Iterable


class NumpyDataset:
    def __init__(self, dataset_filename):
        super().__init__()

        if os.path.isfile(f"{dataset_filename}.pickle"):
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

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, indices: Union[str, Iterable[int]]):
        if isinstance(indices, str):
            return self.dataset[indices]

        return {k: v[indices] for k, v in self.dataset.items()}
