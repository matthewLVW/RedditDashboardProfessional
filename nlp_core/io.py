# nlp_core/io.py

import os
import io
import json
import datetime
from typing import List, Optional
import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq

def zst_to_parquet(zst_path: str, output_dir: str, subs: Optional[List[str]] = None, chunk_size: int = 10000) -> None:
    """
    Decompress a Zstandard .zst Reddit comments dump and write to partitioned Parquet dataset.

    Args:
        zst_path (str): Path to the input .zst file.
        output_dir (str): Root directory for output Parquet files.
        subs (List[str], optional): List of subreddit names to include. If None, include all.
        chunk_size (int): Number of records to process per chunk.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If output_dir is empty or not writable.
    """
    if not os.path.isfile(zst_path):
        raise FileNotFoundError(f"Input file not found: {zst_path}")
    if not output_dir:
        raise ValueError("Output directory must be a non-empty string.")
    os.makedirs(output_dir, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    records = []
    try:
        with open(zst_path, 'rb') as fh:
            stream = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream, encoding='utf-8')
            for line in text_stream:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
                subreddit = data.get("subreddit")
                if subs and subreddit not in subs:
                    continue
                created = data.get("created_utc")
                if created is None:
                    continue
                # Convert epoch to date string for partitioning
                date = datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d")
                data["_date"] = date
                records.append(data)

                if len(records) >= chunk_size:
                    _write_parquet_chunk(records, output_dir)
                    records = []
            # Write remaining records
            if records:
                _write_parquet_chunk(records, output_dir)
    except Exception as e:
        raise RuntimeError(f"Error during ZST to Parquet conversion: {e}")

def _write_parquet_chunk(records: List[dict], output_dir: str) -> None:
    """Helper to write a chunk of records to partitioned Parquet files."""
    try:
        table = pa.Table.from_pylist(records)
        # Partition by subreddit and date
        pq.write_to_dataset(
            table,
            root_path=output_dir,
            partition_cols=["subreddit", "_date"],
            use_dictionary=True,
            existing_data_behavior="overwrite_or_ignore"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to write Parquet chunk: {e}")
