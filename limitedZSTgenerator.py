import zstandard as zstd
import io
import json
from datetime import datetime

INPUT_ZST = "comments/RC_2024-11.zst"   # Path to your input file
OUTPUT_ZST = "tests/data/election_subs_nov1-15.zst"  # Path to new output file

# Your target subreddits (case-sensitive as in the data)
TARGET_SUBS = set([
    "politics", "worldnews", "news", "PresidentialElection", "PoliticalDiscussion",
    "AskALiberal", "moderatepolitics", "neutralpolitics", "WayOfTheBern", "SandersForPresident",
    "AskTrumpSupporters", "EnoughTrumpSpam", "conservative", "libertarian", "Progressive",
    "SocialDemocracy", "Anarchism", "centreistpolitics", "uspolitics", "ukpolitics",
    "canada", "europe", "ukraine", "Coronavirus", "China_Flu", "conspiracy",
    "PoliticalHumor", "PoliticalCompassMemes", "Ask_Politics", "PopheadsGetsPolitical"
])

DATE_START = datetime(2024, 11, 1)
DATE_END = datetime(2024, 11, 15, 23, 59, 59)

kept = 0
with open(INPUT_ZST, "rb") as infile, open(OUTPUT_ZST, "wb") as outfile:
    dctx = zstd.ZstdDecompressor()
    stream = dctx.stream_reader(infile)
    text_stream = io.TextIOWrapper(stream, encoding="utf-8")
    cctx = zstd.ZstdCompressor()
    with cctx.stream_writer(outfile) as writer:
        for line in text_stream:
            try:
                data = json.loads(line)
                subreddit = data.get("subreddit")
                if subreddit not in TARGET_SUBS:
                    continue
                created = data.get("created_utc")
                if created is None:
                    continue
                dt = datetime.utcfromtimestamp(int(created))
                if not (DATE_START <= dt <= DATE_END):
                    continue
                writer.write((json.dumps(data) + "\n").encode("utf-8"))
                kept += 1
                if kept % 10000 == 0:
                    print(f"{kept} comments written...")
            except Exception:
                continue

print(f"Done! {kept} comments extracted to {OUTPUT_ZST}")
