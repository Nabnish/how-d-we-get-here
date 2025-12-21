import re

def clean_medical_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = raw_text

    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    noise_patterns = [
        r"Page\s+\d+\s+of\s+\d+",
        r"Confidential",
        r"Lab Report",
        r"Patient Report"
    ]
    for pat in noise_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    unit_map = {
        r"mg\/dl": "mg/dL",
        r"g\/dl": "g/dL",
        r"cells\/mcL": "cells/mcL"
    }
    for k, v in unit_map.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)

    return text.strip()
