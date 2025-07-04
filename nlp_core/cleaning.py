# nlp_core/cleaning.py

import re
import emoji
from typing import Optional
from langdetect import detect, DetectorFactory

# Fix random seed for langdetect for consistency
DetectorFactory.seed = 0

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')

def clean_text(text: Optional[str]) -> str:
    """
    Clean Reddit comment text by removing URLs, emojis, and non-English posts.

    Args:
        text (str): Input comment body.

    Returns:
        str: Cleaned text (lowercased, no URLs/emojis), or empty string if filtered out.
    """
    if not text or text.strip().lower() in {"[deleted]", "[removed]"}:
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(URL_PATTERN, '', text)
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    # Deduplicate whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Quick language detection: skip non-English texts
    try:
        lang = detect(text)
        if lang != 'en':
            return ""
    except Exception:
        # If detection fails, keep text by default
        pass

    return text
