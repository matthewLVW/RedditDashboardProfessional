# tests/test_io.py

import os
import pandas as pd
import pytest
from nlp_core.io import zst_to_parquet

@pytest.fixture
def sample_zst(tmp_path):
    # Create a small .zst with JSON lines
    import json, zstandard
    data = [
        {"subreddit": "politics", "created_utc": 1700000000, "body": "Test comment"}
    ]
    zst_path = tmp_path / "test.zst"
    cctx = zstandard.ZstdCompressor()
    with open(zst_path, "wb") as f:
        f.write(cctx.compress((json.dumps(data[0])+"\n").encode('utf-8')))
    return str(zst_path), str(tmp_path)

def test_zst_to_parquet_creates_files(sample_zst):
    zst_path, out_dir = sample_zst
    zst_to_parquet(zst_path, out_dir, subs=["politics"])
    # Check that parquet files exist
    part_path = os.path.join(out_dir, "subreddit=politics")
    assert os.path.isdir(part_path)
    files = [f for f in os.listdir(out_dir) if f.endswith(".parquet") or f.endswith(".parquet.dataset")]
    assert files or os.listdir(part_path)
