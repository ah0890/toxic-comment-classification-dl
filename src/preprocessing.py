"""Text-cleaning utilities shared by the notebook and the prediction script.

The same ``clean_text`` function must be used for training data and for new
comments at prediction time, so the model always sees text in the same format.
"""
import html
import re

import pandas as pd

# Pre-compiled regular expressions (compiled once, reused for every comment)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
APOSTROPHE_PATTERN = re.compile(r"['’`]")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text):
    """Clean one comment and return the cleaned string.

    Steps, in order:
      1. Missing values (None / NaN) become an empty string.
      2. Convert to lowercase.
      3. Decode HTML entities (``&amp;`` -> ``&``) and remove HTML tags.
      4. Remove URLs and e-mail addresses.
      5. Remove apostrophes so contractions stay one word (``don't`` -> ``dont``).
      6. Remove everything that is not a letter or whitespace: punctuation,
         digits, special characters, and emojis.
      7. Collapse repeated whitespace and trim the ends.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    text = str(text).lower()
    text = html.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = APOSTROPHE_PATTERN.sub("", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()
