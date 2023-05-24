from typing import Mapping, List
import pyarrow.parquet as pq

import smart_open


class ParquetStreamer:

    def __init__(self, parquet_url: str, observed_rows: List[str], headers: Mapping[str, str] = None):
        self.parquet_url = parquet_url
        self.headers = headers
        self.observed_rows = observed_rows

        fp = smart_open.open(self.parquet_url, "rb", transport_params={'headers': self.headers if self.headers else {}})
        self.parquet_file = pq.ParquetFile(fp)

    def __iter__(self):
        num_row_groups = self.parquet_file.num_row_groups

        for row_group_index in range(num_row_groups):
            row_group = self.parquet_file.read_row_group(row_group_index)
            columns = {column_name: row_group.column(column_name) for column_name in self.observed_rows}

            for row_index in range(row_group.num_rows):
                row = {column_name: column[row_index].as_py() for column_name, column in columns.items()}
                yield row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.parquet_file.close()
