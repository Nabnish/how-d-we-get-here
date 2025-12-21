import re
from typing import Dict, List

TEST_NAME_MAP = {
    "hb": "Hemoglobin",
    "hemoglobin": "Hemoglobin",
    "wbc": "WBC Count",
    "rbc": "RBC Count",
    "platelet": "Platelet Count",
    "glucose": "Blood Glucose"
}

def normalize_test_name(name: str) -> str:
    key = name.lower().strip()
    return TEST_NAME_MAP.get(key, name.title())

def detect_status(value: float, ref_range: str) -> str:
    try:
        low, high = ref_range.replace("–", "-").split("-")
        low, high = float(low.strip()), float(high.strip())
        if value < low:
            return "Low"
        if value > high:
            return "High"
        return "Normal"
    except:
        return "Unknown"

def extract_lab_results(clean_text: str) -> Dict:
    lab_results: List[Dict] = []

    pattern = re.compile(
        r"([A-Za-z\s]+)\s*[:\-]?\s*([\d\.]+)\s*([A-Za-z\/%]+)?\s*\(?([\d\.\-\–]+)?\)?",
        re.IGNORECASE
    )

    for match in pattern.finditer(clean_text):
        test_raw, value, unit, ref = match.groups()

        try:
            value = float(value)
        except:
            continue

        test_name = normalize_test_name(test_raw)
        ref_range = ref if ref else "Unknown"
        status = detect_status(value, ref_range)

        lab_results.append({
            "test_name": test_name,
            "value": value,
            "unit": unit or "",
            "reference_range": ref_range,
            "status": status
        })

    return {
        "lab_results": lab_results,
        "total_tests": len(lab_results)
    }
