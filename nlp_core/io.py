# nlp_core/io.py

import os
import io
import json
import datetime
from typing import List, Optional
import zstandard as zstd
import pyarrow as pa
import pyarrow.parquet as pq

def zst_to_parquet(
    zst_path: str,
    output_dir: str,
    subs: Optional[List[str]] = None,
    date_min: Optional[str] = None,   # format: "YYYY-MM-DD"
    date_max: Optional[str] = None,   # format: "YYYY-MM-DD"
    chunk_size: int = 10000
) -> None:
    """
    Decompress a Zstandard .zst Reddit comments dump and write to partitioned Parquet dataset.

    Args:
        zst_path (str): Path to the input .zst file.
        output_dir (str): Root directory for output Parquet files.
        subs (List[str], optional): List of subreddit names to include. If None, include all.
        date_min (str, optional): Minimum date ("YYYY-MM-DD") to include.
        date_max (str, optional): Maximum date ("YYYY-MM-DD") to include.
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

    # Prepare date boundaries
    dt_min = datetime.date.fromisoformat(date_min) if date_min else None
    dt_max = datetime.date.fromisoformat(date_max) if date_max else None

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
                    continue
                 # ─── QUICK FIX: normalise the “edited” field ──────────────────────
                edited_val = data.get("edited")
                if isinstance(edited_val, bool):          # False → never edited
                    data["edited"] = None                 # use None instead of False
    # ──────────────────────────────────────────────────────────────────

                subreddit = data.get("subreddit")
                if subs and subreddit not in subs:
                    continue
                created = data.get("created_utc")
                if created is None:
                    continue
                # Convert epoch to date string for partitioning and filtering
                date_str = datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d")
                date_obj = datetime.date.fromisoformat(date_str)
                if (dt_min and date_obj < dt_min) or (dt_max and date_obj > dt_max):
                    continue
                data["_date"] = date_str
                records.append(data)

                if len(records) >= chunk_size:
                    _write_parquet_chunk(records, output_dir)
                    records = []
            if records:
                _write_parquet_chunk(records, output_dir)
    except Exception as e:
        raise RuntimeError(f"Error during ZST to Parquet conversion: {e}")
import pyarrow.types as pat
def _write_parquet_chunk(records: list[dict], output_dir: str) -> None:
    try:
        table = pa.Table.from_pylist(records)

        # ─── Strip struct<> columns (empty dicts) ──────────────────────────
        empty_struct_cols = [
            field.name
            for field in table.schema
            if pat.is_struct(field.type) and len(field.type) == 0
        ]
        if empty_struct_cols:
            table = table.drop(empty_struct_cols)

        # Partition by subreddit and _date
        pq.write_to_dataset(
            table,
            root_path=output_dir,
            partition_cols=["subreddit", "_date"],
            use_dictionary=True,
            existing_data_behavior="overwrite_or_ignore",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to write Parquet chunk: {e}")
