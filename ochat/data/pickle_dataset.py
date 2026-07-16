"""
Convert parquet dataset into pickle

Usage: python -m ochat.data.pickle_dataset --data-prefix pretokenized_data
"""

import pyarrow.parquet as pq
import os
import orjson
import pickle
import argparse
from typing import Dict, Any
from pydantic import BaseModel, Field


class DataArguments(BaseModel):
    data_prefix: str = Field(...)


def make_dict(dataset_filename: str) -> Dict[str, Any]:

    table = pq.read_table(dataset_filename, memory_map=True)
    final_data = {}
    final_data['dataset'] = {
        k: v.to_numpy() for k, v in zip(table.column_names, table.columns)
    }
    final_data['length'] = table.num_rows
    final_data['metadata'] = (table.schema.metadata or {}).get(b"metadata_json", None)
    if final_data['metadata'] is not None:
        final_data['metadata'] = orjson.loads(final_data['metadata'])
    
    return final_data


def main(args: DataArguments):

    for split in ['train', 'eval']:
        result_filename = f"{args.data_prefix}.{split}.pickle"
        filename = f"{args.data_prefix}.{split}.parquet"
        if os.path.isfile(result_filename):
            print(f'File {result_filename} already exists. Skipping {filename}...')
            continue
        if not os.path.isfile(filename):
            print(f'There is no file named {filename}! Checking probable parts...')
            index = 0
            while True:
                result_filename = f"{args.data_prefix}.{split}.part{index:03d}.pickle"
                filename = f"{args.data_prefix}.{split}.part{index:03d}.parquet"
                if os.path.isfile(result_filename):
                    print(f'File {result_filename} already exists. Skipping {filename}...')
                    index += 1
                    continue
                if not os.path.isfile(filename):
                    if index == 0:
                        print(f'There is no file named {filename}! Skipping...')
                    break
                print(f'Read file {filename}...')
                data = make_dict(filename)
                print(f'Pickling to file {result_filename}...')
                with open(result_filename, 'wb') as f:
                    pickle.dump(data, f)
                index += 1
            continue
        print(f'Read file {filename}...')
        data = make_dict(filename)
        print(f'Pickling to file {result_filename}...')
        with open(result_filename, 'wb') as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-prefix', type=str, required=True)
    args = parser.parse_args()

    args = DataArguments(**vars(args))

    main(args)