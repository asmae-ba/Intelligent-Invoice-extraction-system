"""
verify_extraction.py

Rule-based verification for extracted invoice fields.
This checks whether COMPANY, ADDRESS, DATE, and TOTAL look valid.

Usage:
    python verify_extraction.py --input prediction_output.json

It can read either:
    {"fields": {"company": "...", "date": "...", "total": "..."}}
Or directly:
    {"company": "...", "date": "...", "total": "..."}
"""

import argparse
import json
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple


DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%m.%d.%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
]


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip()


def normalize_amount(value: Optional[str]) -> Tuple[Optional[float], str]:
    """
    Extracts a numeric amount from text like '$1,250.50' or 'RM 12.00'.
    """
    raw = clean_text(value)
    if not raw:
        return None, ""

    # Keep digits, commas, dots, and minus sign.
    candidates = re.findall(r"-?\d[\d,]*(?:\.\d+)?", raw)
    if not candidates:
        return None, raw

    # Usually the amount is the last/longest numeric candidate.
    candidate = max(candidates, key=len)
    normalized = candidate.replace(",", "")

    try:
        return float(normalized), candidate
    except ValueError:
        return None, candidate


def parse_date(value: Optional[str]) -> Tuple[Optional[date], str]:
    raw = clean_text(value)
    if not raw:
        return None, ""

    # Remove common labels if OCR includes them.
    raw = re.sub(r"(?i)\b(date|invoice date|issued date)\b[:\-]*", "", raw).strip()
    raw = raw.replace(",", " ")
    raw = " ".join(raw.split())

    # Try direct formats.
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date(), raw
        except ValueError:
            pass

    # Try extracting date-like substring.
    patterns = [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        r"\d{1,2}[.]\d{1,2}[.]\d{2,4}",
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
        r"[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4}",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            text = match.group(0)
            for fmt in DATE_FORMATS:
                try:
                    return datetime.strptime(text, fmt).date(), text
                except ValueError:
                    pass

    return None, raw


def validate_company(value: Optional[str]) -> Dict:
    text = clean_text(value)
    if not text:
        return {"status": "invalid", "message": "Company/vendor name is missing."}
    if len(text) < 2:
        return {"status": "warning", "message": "Company/vendor name is too short."}
    if not re.search(r"[A-Za-z0-9]", text):
        return {"status": "invalid", "message": "Company/vendor name has no readable characters."}
    return {"status": "valid", "message": "Company/vendor name detected."}


def validate_address(value: Optional[str]) -> Dict:
    text = clean_text(value)
    if not text:
        return {"status": "warning", "message": "Address is missing or not detected."}
    if len(text) < 5:
        return {"status": "warning", "message": "Address looks too short."}
    return {"status": "valid", "message": "Address detected."}


def validate_date(value: Optional[str]) -> Dict:
    parsed, detected_text = parse_date(value)
    if parsed is None:
        return {"status": "invalid", "message": "Date is missing or invalid.", "normalized": None}

    if parsed.year < 1900:
        return {"status": "invalid", "message": "Date year is too old.", "normalized": parsed.isoformat()}

    today = date.today()
    if parsed > today:
        return {"status": "warning", "message": "Date is in the future.", "normalized": parsed.isoformat()}

    return {
        "status": "valid",
        "message": f"Valid date detected from '{detected_text}'.",
        "normalized": parsed.isoformat(),
    }


def validate_total(value: Optional[str]) -> Dict:
    amount, detected_text = normalize_amount(value)
    if amount is None:
        return {"status": "invalid", "message": "Total amount is missing or invalid.", "normalized": None}

    if amount < 0:
        return {"status": "invalid", "message": "Total amount is negative.", "normalized": amount}

    if amount == 0:
        return {"status": "warning", "message": "Total amount is zero.", "normalized": amount}

    return {
        "status": "valid",
        "message": f"Valid total amount detected from '{detected_text}'.",
        "normalized": amount,
    }


def verify_extraction(fields: Dict) -> Dict:
    """
    Verifies extracted fields.

    Expected keys:
        company, address, date, total
    """
    company = fields.get("company") or fields.get("vendor") or fields.get("vendor_name")
    address = fields.get("address")
    invoice_date = fields.get("date") or fields.get("invoice_date")
    total = fields.get("total") or fields.get("total_amount")

    checks = {
        "company": validate_company(company),
        "address": validate_address(address),
        "date": validate_date(invoice_date),
        "total": validate_total(total),
    }

    missing_fields = []
    invalid_fields = []
    warning_fields = []

    for field_name, check in checks.items():
        if check["status"] == "invalid":
            invalid_fields.append(field_name)
        elif check["status"] == "warning":
            warning_fields.append(field_name)

    required_fields = ["company", "date", "total"]
    for field_name in required_fields:
        value = {
            "company": company,
            "date": invoice_date,
            "total": total,
        }.get(field_name)
        if not clean_text(value):
            missing_fields.append(field_name)

    if invalid_fields:
        overall_status = "needs_review"
    elif warning_fields:
        overall_status = "valid_with_warnings"
    else:
        overall_status = "valid"

    return {
        "overall_status": overall_status,
        "checks": checks,
        "missing_fields": missing_fields,
        "invalid_fields": invalid_fields,
        "warning_fields": warning_fields,
    }


def load_fields_from_json(input_path: str) -> Dict:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "fields" in data:
        return data["fields"]

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify extracted invoice fields.")
    parser.add_argument("--input", required=True, help="Prediction JSON file")
    parser.add_argument("--output", default="", help="Optional verification output JSON")
    args = parser.parse_args()

    fields = load_fields_from_json(args.input)
    verification = verify_extraction(fields)

    print(json.dumps(verification, indent=2, ensure_ascii=False))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(verification, f, indent=2, ensure_ascii=False)
        print(f"Saved verification result to: {args.output}")


if __name__ == "__main__":
    main()
