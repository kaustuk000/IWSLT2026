import re
import unicodedata
import numpy as np
import pandas as pd
import os


def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\u00ad]", "", text)
    text = text.replace("\u0929", "\u0928\u093C")
    text = text.replace("\u0931", "\u0930\u093C")
    text = text.replace("\u0934", "\u0933\u093C")
    text = re.sub(
        r"[^\u0900-\u097F"   # Devanagari block
        r"\u0966-\u096F"     # Devanagari digits ०-९
        r"a-zA-Z"            # Latin letters (code-mix, proper nouns)
        r"0-9"               # ASCII digits
        r"\s।?,!.\-\'\"]+",
        "",
        text,
    )
    return text.strip()


