from io import BytesIO
from typing import Mapping, List, Callable
import pyarrow.parquet as pq
import requests

import smart_open


class ParquetStreamer:

    def __init__(self, parquet_url: str, observed_columns: List[str], headers: Mapping[str, str] = None):
        self.parquet_url = parquet_url
        self.headers = headers
        self.observed_columns = observed_columns

        fp = smart_open.open(self.parquet_url, "rb", transport_params={'headers': self.headers if self.headers else {}})
        parquet_dataset = pq.ParquetDataset(fp)

        self.num_rows = sum(p.count_rows() for p in parquet_dataset.fragments)

        self.parquet_file = pq.ParquetFile(fp)

    def __iter__(self):
        num_row_groups = self.parquet_file.num_row_groups

        for row_group_index in range(num_row_groups):
            row_group = self.parquet_file.read_row_group(row_group_index)
            columns = {column_name: row_group.column(column_name) for column_name in self.observed_columns}

            for row_index in range(row_group.num_rows):
                row = {column_name: column[row_index].as_py() for column_name, column in columns.items()}
                yield row

    def __len__(self):
        return self.num_rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.parquet_file.close()


class FileWrappingParquetIterator:

    def __init__(self, fp, observed_columns: List[str]):
        self.observed_columns = observed_columns

        parquet_dataset = pq.ParquetDataset(fp)

        self.num_rows = sum(p.count_rows() for p in parquet_dataset.fragments)

        self.parquet_file = pq.ParquetFile(fp)

    def __iter__(self):
        num_row_groups = self.parquet_file.num_row_groups

        for row_group_index in range(num_row_groups):
            row_group = self.parquet_file.read_row_group(row_group_index)
            columns = {column_name: row_group.column(column_name) for column_name in self.observed_columns}

            for row_index in range(row_group.num_rows):
                row = {column_name: column[row_index].as_py() for column_name, column in columns.items()}
                yield row

    def __len__(self):
        return self.num_rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.parquet_file.close()
