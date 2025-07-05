# scripts/run_pipeline.py

import os
import dask.dataframe as dd
from dask.distributed import Client
from nlp_core.io import zst_to_parquet
from nlp_core.cleaning import clean_text
from nlp_core.spacy_pipe import process_texts, analyze_text
from nlp_core.sentiment import fused_sentiment
from nlp_core.emotion import detect_emotions
from nlp_core.stance import detect_stance
from nlp_core.topic import get_topics, extract_keywords
from nlp_core.polarization import js_divergence
from nlp_core.sarcasm import detect_sarcasm

def main():
    # Initialize Dask client for parallel processing
    client = Client()
    
    # Paths and configuration
    zst_path = "data/raw/RC_2024-11.zst"
    parquet_dir = "data/parquet"
    clean_dir = "data/parquet_clean"
    
    # Step 1: Convert raw ZST to Parquet by subreddit
    try:
        zst_to_parquet(zst_path, parquet_dir, subs=[
            "politics", "conservative", "moderatepolitics", "politicaldiscussion"
        ])
    except Exception as e:
        print(f"Data ingestion failed: {e}")
        return

    # Step 2: Load Parquet with Dask
    try:
        df = dd.read_parquet(parquet_dir)
    except Exception as e:
        print(f"Failed to read parquet data: {e}")
        return

    # Step 3: Clean text
    df = df.dropna(subset=["body"])
    df["clean_body"] = df["body"].map(clean_text, meta=("clean_body", "object"))
    df = df[df["clean_body"] != ""]

    # Persist cleaned data
    df.to_parquet(clean_dir, overwrite=True)

    # Step 4: NLP Feature Extraction (example on sample partition)
    try:
        sample_df = df.head(1000)  # process a sample in-memory for demonstration
    except Exception as e:
        print(f"Failed to take sample: {e}")
        sample_df = df.compute().head(1000)

    texts = sample_df["clean_body"].tolist()
    docs = process_texts(texts)
    nlp_features = [analyze_text(doc) for doc in docs]
    sample_df["entities"] = [features["entities"] for features in nlp_features]
    sample_df["pos_counts"] = [features["pos_counts"] for features in nlp_features]

    # Sentiment
    sample_df["sentiment"] = sample_df["clean_body"].apply(fused_sentiment, meta=("sentiment", "float"))

    # Emotion
    sample_df["emotions"] = sample_df["clean_body"].apply(detect_emotions, meta=("emotions", "object"))

    # Stance
    sample_df["stance"] = sample_df["clean_body"].apply(detect_stance, meta=("stance", "object"))

    # Sarcasm
    sample_df["sarcasm_score"] = sample_df["clean_body"].apply(detect_sarcasm, meta=("sarcasm_score", "float"))

    # Topic modeling (batch example)
    try:
        topics, probs = get_topics(texts)
        sample_df["topic"] = topics
        confidences = []
        for p in probs:
            if p is None:
                confidences.append(0.0)
            elif hasattr(p, 'max'):
                try:
                    confidences.append(float(p.max()))
                except Exception:
                    confidences.append(float(p))
            else:
                try:
                    confidences.append(float(max(p)))
                except Exception:
                    confidences.append(float(p))
        sample_df["topic_confidence"] = confidences
    except Exception as e:
        print(f"Topic modeling failed: {e}")

    # Step 5: Save features to disk
    features_dir = "data/features"
    os.makedirs(features_dir, exist_ok=True)
    dd.from_pandas(sample_df, npartitions=1).to_parquet(features_dir, overwrite=True)

    client.close()

if __name__ == "__main__":
    main()
