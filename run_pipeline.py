# scripts/run_pipeline.py
"""
End-to-end Reddit pipeline:
    Zstandard dump  ➜  partitioned Parquet  ➜  cleaned & NLP-enriched Parquet

Run:
    python -m scripts.run_pipeline \
        --zst data/raw/RC_2024-11.zst \
        --out-root data   \
        --sub politics conservative moderatepolitics politicaldiscussion \
        --workers 8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import dask.dataframe as dd
from dask.distributed import Client, performance_report

from nlp_core.io import zst_to_parquet
from nlp_core.cleaning import clean_text
from nlp_core.spacy_pipe import process_texts, analyze_text
from nlp_core.sentiment import fused_sentiment
from nlp_core.emotion import detect_emotions
from nlp_core.stance import detect_stance
from nlp_core.topic import get_topics, extract_keywords
from nlp_core.polarization import js_divergence
from nlp_core.sarcasm import detect_sarcasm


# ─────────────────────────────── helpers ────────────────────────────────
def _nlp_partition(df_part):
    """Apply all NLP functions to a pandas partition and return the enriched df."""
    # Spacy bulk processing – single call per partition keeps it fast
    docs = process_texts(df_part["clean_body"].tolist())
    feats = [analyze_text(doc) for doc in docs]

    df_part = df_part.copy()
    df_part["entities"] = [f["entities"] for f in feats]
    df_part["pos_counts"] = [f["pos_counts"] for f in feats]
    df_part["sentiment"] = df_part["clean_body"].apply(fused_sentiment)
    df_part["emotions"] = df_part["clean_body"].apply(detect_emotions)
    df_part["stance"] = df_part["clean_body"].apply(detect_stance)
    df_part["sarcasm_score"] = df_part["clean_body"].apply(detect_sarcasm)

    # Topic modelling: run only if partition is non-empty
    texts = df_part["clean_body"].tolist()
    if texts:
        topics, probs = get_topics(texts)
        df_part["topic"] = topics
        df_part["topic_confidence"] = [float(p.max()) if hasattr(p, "max") else float(p) for p in probs]
        # keywords once per partition for illustration
        df_part.attrs["keywords"] = extract_keywords()
    else:
        df_part["topic"] = []
        df_part["topic_confidence"] = []
    return df_part


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zst", required=True, help="Path to compressed comments dump")
    p.add_argument("--out-root", required=True, help="Root folder for all artefacts")
    p.add_argument("--sub", nargs="+", default=None, help="Subreddits to include")
    p.add_argument("--workers", type=int, default=4, help="Dask workers")
    p.add_argument("--chunk", type=int, default=50_000, help="Rows per Arrow chunk")
    return p.parse_args()


# ─────────────────────────────── pipeline ───────────────────────────────
def main():
    args = parse_args()
    ROOT = Path(args.out_root).resolve()
    parquet_raw = ROOT / "parquet_raw"
    parquet_clean = ROOT / "parquet_clean"
    parquet_feat = ROOT / "features"

    # 1 ─── ZST ➜ Parquet
    print("▪ Converting ZST to Parquet …")
    zst_to_parquet(
        args.zst,
        str(parquet_raw),
        subs=args.sub,
        chunk_size=args.chunk,
    )

    # 2 ─── Spin up Dask cluster
    client = Client(n_workers=args.workers, threads_per_worker=1)
    print(f"▪ Dask dashboard: {client.dashboard_link}")

    with performance_report(filename="dask_report.html"):
        # 3 ─── Load & clean
        df = dd.read_parquet(parquet_raw)
        df = df.dropna(subset=["body"])
        df["clean_body"] = df["body"].map(clean_text, meta=("clean_body", "object"))
        df = df[df["clean_body"] != ""]
        df = df.persist()
        df.to_parquet(parquet_clean, overwrite=True)
        print("  ↳ Cleaned dataframe persisted.")

        # 4 ─── NLP enrichment (map_partitions keeps memory bounded)
        df_enriched = df.map_partitions(_nlp_partition, meta=df._meta)
        df_enriched.to_parquet(parquet_feat, overwrite=True)
        client.wait_for_workers(1)   # ensure tasks were scheduled

    print("✓ Pipeline completed. Outputs:")
    print(f"   raw    ➜ {parquet_raw}")
    print(f"   clean  ➜ {parquet_clean}")
    print(f"   feats  ➜ {parquet_feat}")
    client.close()


if __name__ == "__main__":
    main()
