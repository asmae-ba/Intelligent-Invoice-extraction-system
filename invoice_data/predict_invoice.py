"""
predict_invoice.py

Loads your trained LayoutLM model and predicts invoice fields from a real invoice image/PDF.

This script works with the model trained by your training file:
    ./layoutlm_invoice_model

Your current labels are:
    COMPANY, ADDRESS, DATE, TOTAL

Usage:
    python predict_invoice.py --input samples/invoice.jpg --model-dir ./layoutlm_invoice_model --output result.json
    python predict_invoice.py --input samples/invoice.pdf --model-dir ./layoutlm_invoice_model --output result.json

Required packages:
    pip install torch transformers pillow pytesseract scikit-learn

For PDF input:
    pip install pdf2image
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

from ocr_to_layoutlm import run_tesseract_ocr
from verify_extraction import verify_extraction


DEFAULT_LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-DATE",
    "I-DATE",
    "B-TOTAL",
    "I-TOTAL",
]

MAX_LENGTH = 512
BASE_TOKENIZER = "microsoft/layoutlm-base-uncased"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_dir: str) -> LayoutLMTokenizerFast:
    """
    Tries to load tokenizer from model_dir.
    If tokenizer was not saved there, loads the original LayoutLM tokenizer.
    """
    try:
        return LayoutLMTokenizerFast.from_pretrained(model_dir)
    except Exception:
        return LayoutLMTokenizerFast.from_pretrained(BASE_TOKENIZER)


def load_model_and_tokenizer(model_dir: str):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Make sure your training script saved the model to ./layoutlm_invoice_model"
        )

    tokenizer = load_tokenizer(model_dir)
    model = LayoutLMForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def get_id2label(model) -> Dict[int, str]:
    """
    Reads id2label from model config.
    Falls back to DEFAULT_LABELS if needed.
    """
    raw = getattr(model.config, "id2label", None)
    if raw:
        id2label = {}
        for key, value in raw.items():
            try:
                id2label[int(key)] = value
            except Exception:
                id2label[key] = value
        return id2label

    return {i: label for i, label in enumerate(DEFAULT_LABELS)}


def align_boxes_to_tokens(word_ids: List[int], boxes: List[List[int]]) -> torch.Tensor:
    """
    Creates a bbox tensor aligned with tokenized input.
    Special tokens get [0,0,0,0].
    """
    aligned_boxes = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_boxes.append(boxes[word_idx])
    return torch.tensor([aligned_boxes], dtype=torch.long)


def predict_word_labels(
    words: List[str],
    boxes: List[List[int]],
    model,
    tokenizer,
    device: torch.device,
) -> List[Dict]:
    """
    Predicts one BIO label per OCR word.
    Uses the first sub-token prediction for each original word.
    """
    if not words:
        return []

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    word_ids = encoding.word_ids(batch_index=0)
    bbox_tensor = align_boxes_to_tokens(word_ids, boxes)

    encoding = {k: v.to(device) for k, v in encoding.items()}
    bbox_tensor = bbox_tensor.to(device)

    model.to(device)

    with torch.no_grad():
        outputs = model(**encoding, bbox=bbox_tensor)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()
        pred_ids = probs.argmax(dim=-1).tolist()

    id2label = get_id2label(model)

    word_results: List[Dict] = []
    seen_words = set()

    for token_index, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx in seen_words:
            continue
        if word_idx >= len(words):
            continue

        pred_id = pred_ids[token_index]
        label = id2label.get(pred_id, "O")
        confidence = float(probs[token_index][pred_id])

        word_results.append(
            {
                "word_index": word_idx,
                "word": words[word_idx],
                "box": boxes[word_idx],
                "label": label,
                "confidence": round(confidence, 4),
            }
        )
        seen_words.add(word_idx)

    return word_results


def label_to_entity(label: str) -> Tuple[str, str]:
    """
    Converts B-COMPANY into (B, COMPANY).
    """
    if label == "O" or "-" not in label:
        return "O", "O"
    prefix, entity = label.split("-", 1)
    return prefix, entity


def merge_boxes(boxes: List[List[int]]) -> List[int]:
    if not boxes:
        return [0, 0, 0, 0]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def build_spans(word_results: List[Dict]) -> List[Dict]:
    """
    Builds entity spans from BIO token labels.
    """
    spans = []
    current = None

    for item in word_results:
        label = item["label"]
        prefix, entity = label_to_entity(label)

        if prefix == "O":
            if current:
                spans.append(current)
                current = None
            continue

        if prefix == "B" or current is None or current["entity"] != entity:
            if current:
                spans.append(current)
            current = {
                "entity": entity,
                "words": [item["word"]],
                "boxes": [item["box"]],
                "confidences": [item["confidence"]],
            }
        else:
            current["words"].append(item["word"])
            current["boxes"].append(item["box"])
            current["confidences"].append(item["confidence"])

    if current:
        spans.append(current)

    clean_spans = []
    for span in spans:
        text = " ".join(span["words"]).strip()
        clean_spans.append(
            {
                "entity": span["entity"],
                "text": text,
                "box": merge_boxes(span["boxes"]),
                "confidence": round(sum(span["confidences"]) / max(1, len(span["confidences"])), 4),
            }
        )

    return clean_spans


def looks_like_amount(text: str) -> bool:
    import re
    return bool(re.search(r"\d+[,.]?\d*", text))


def looks_like_date(text: str) -> bool:
    import re
    patterns = [
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        r"\d{1,2}[.]\d{1,2}[.]\d{2,4}",
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
        r"[A-Za-z]{3,9}\s+\d{1,2}\s+\d{4}",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def choose_best_span(spans: List[Dict], entity: str) -> str:
    """
    Chooses the best span for each field.
    """
    candidates = [s for s in spans if s["entity"] == entity and s["text"].strip()]
    if not candidates:
        return ""

    if entity == "TOTAL":
        amount_candidates = [s for s in candidates if looks_like_amount(s["text"])]
        if amount_candidates:
            return max(amount_candidates, key=lambda s: (s["confidence"], len(s["text"]))) ["text"]

    if entity == "DATE":
        date_candidates = [s for s in candidates if looks_like_date(s["text"])]
        if date_candidates:
            return max(date_candidates, key=lambda s: (s["confidence"], len(s["text"]))) ["text"]

    # For company/address, longer text with good confidence is usually better.
    return max(candidates, key=lambda s: (len(s["text"]), s["confidence"]))["text"]


def reconstruct_fields(spans: List[Dict]) -> Dict[str, str]:
    """
    Converts entity spans into final invoice fields.
    """
    return {
        "company": choose_best_span(spans, "COMPANY"),
        "address": choose_best_span(spans, "ADDRESS"),
        "date": choose_best_span(spans, "DATE"),
        "total": choose_best_span(spans, "TOTAL"),
    }


def save_json(data: Dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(fields: Dict[str, str], verification: Dict, output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value", "status", "message"])
        checks = verification.get("checks", {})
        for field_name, value in fields.items():
            check = checks.get(field_name, {})
            writer.writerow([
                field_name,
                value,
                check.get("status", ""),
                check.get("message", ""),
            ])

import re


def fallback_extract_from_tokens(tokens):
    import re

    fields = {
        "company": "",
        "address": "",
        "date": "",
        "total": ""
    }

    # =====================================================
    # 1. Clean OCR tokens
    # =====================================================
    clean_tokens = []

    for t in tokens:
        word = str(t.get("word", "")).strip()
        box = t.get("box", [0, 0, 0, 0])

        if not word:
            continue

        # Remove noisy OCR from very top image border
        if box[1] <= 10:
            continue

        # Remove very small noise
        if len(word) <= 1 and not word.isdigit():
            continue

        clean_tokens.append({
            "word": word,
            "box": box
        })

    # =====================================================
    # 2. Group tokens into OCR text lines
    # =====================================================
    lines = []

    for token in clean_tokens:
        box = token["box"]
        y_center = (box[1] + box[3]) / 2

        placed = False

        for line in lines:
            # Smaller threshold prevents wrong line merging
            if abs(line["y_center"] - y_center) <= 8:
                line["tokens"].append(token)
                line["y_center"] = (line["y_center"] + y_center) / 2
                placed = True
                break

        if not placed:
            lines.append({
                "y_center": y_center,
                "tokens": [token]
            })

    lines = sorted(lines, key=lambda x: x["y_center"])

    line_texts = []

    for line in lines:
        sorted_tokens = sorted(line["tokens"], key=lambda x: x["box"][0])
        text = " ".join(t["word"] for t in sorted_tokens)
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            line_texts.append({
                "text": text,
                "y": line["y_center"]
            })

    full_text = " ".join(line["text"] for line in line_texts)

    # Helper function: safer keyword matching
    def has_keyword(text, keyword):
        text_upper = text.upper()
        keyword_upper = keyword.upper()

        # Short keywords must match as whole words
        if len(keyword_upper) <= 3:
            return re.search(r"\b" + re.escape(keyword_upper) + r"\b", text_upper) is not None

        return keyword_upper in text_upper

    # =====================================================
    # 3. DATE extraction
    # Supports:
    # 15/01/2019
    # 23/03/17
    # 2026-06-26
    # 26 Mar 2018
    # =====================================================
    date_match = re.search(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
        r"\b\d{4}-\d{2}-\d{2}\b|"
        r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
        full_text,
        re.IGNORECASE
    )

    if date_match:
        date_value = date_match.group(0)

        # Convert 2-digit year to 4-digit year
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2}$", date_value):
            day, month, year = date_value.split("/")
            year_int = int(year)

            if year_int <= 50:
                year = "20" + year
            else:
                year = "19" + year

            date_value = f"{day}/{month}/{year}"

        fields["date"] = date_value
    # =====================================================
    # 4. TOTAL extraction
    # Avoid GST, TAX, GROSS, SUB-TOTAL, CHANGE, SUMMARY
    # =====================================================
    gst_summary_y = None

    for line in line_texts:
        if "GST SUMMARY" in line["text"].upper():
            gst_summary_y = line["y"]
            break

    total_candidates = []

    for line in line_texts:
        text = line["text"]
        upper = text.upper()

        # Ignore totals inside GST summary section
        if gst_summary_y is not None and line["y"] > gst_summary_y:
            continue

        # Skip wrong total-like lines
        if any(skip in upper for skip in [
            "GST",
            "TAX",
            "GROSS",
            "SUB-TOTAL",
            "TOTAL QTY",
            "LOYALTY",
            "CHANGE",
            "SUMMARY",
            "ROUNDING"
        ]):
            continue

        # Prefer real TOTAL line
        if upper.startswith("TOTAL") or re.search(r"\bTOTAL\b", upper):
            amounts = re.findall(r"\b\d+[.,]\d{2}\b", text)

            if amounts:
                total_candidates.append({
                    "amount": amounts[-1],
                    "y": line["y"],
                    "text": text
                })

    if total_candidates:
        fields["total"] = total_candidates[-1]["amount"].replace(",", ".")
    else:
        # Fallback: use CASH amount if TOTAL is not clear
        cash_candidates = []

        for line in line_texts:
            text = line["text"]
            upper = text.upper()

            if gst_summary_y is not None and line["y"] > gst_summary_y:
                continue

            if "CASH" in upper:
                amounts = re.findall(r"\b\d+[.,]\d{2}\b", text)

                if amounts:
                    cash_candidates.append(amounts[-1])

        if cash_candidates:
            fields["total"] = cash_candidates[0].replace(",", ".")
        else:
            # Final fallback: choose the largest reasonable amount
            all_amounts = re.findall(r"\b\d+[.,]\d{2}\b", full_text)

            if all_amounts:
                numeric_amounts = []

                for amount in all_amounts:
                    try:
                        numeric_amounts.append(float(amount.replace(",", ".")))
                    except ValueError:
                        pass

                if numeric_amounts:
                    fields["total"] = f"{max(numeric_amounts):.2f}"

    # =====================================================
    # 5. COMPANY extraction
    # =====================================================
    stop_company_words = [
        "COPY",
        "RECEIPT",
        "INVOICE",
        "TAX",
        "CASH",
        "DATE",
        "TIME",
        "TEL",
        "TELEPHONE",
        "FAX",
        "EMAIL",
        "GST",
        "ROC",
        "COMPANY NO",
        "NO:",
        "NO.",
        "ADDRESS",
        "ITEM",
        "QTY",
        "PRICE",
        "AMOUNT",
        "TOTAL",
        "SUB-TOTAL",
        "CHANGE",
        "SITE",
        "PRE-AUTHORISATION",
        "PRE-AUTHORIZATION"
    ]

    company_keywords = [
        "SDN BHD",
        "LTD",
        "LIMITED",
        "COMPANY",
        "ENTERPRISE",
        "TRADING",
        "STATIONERY",
        "BOOKS",
        "MARKETING"
    ]

    # Strong first clean top-line rule
    for line in line_texts[:6]:
        text = line["text"].strip()
        upper = text.upper()

        if len(text) < 4:
            continue

        if any(stop in upper for stop in stop_company_words):
            continue

        letters = re.findall(r"[A-Za-z]", text)
        digits = re.findall(r"\d", text)

        if len(letters) >= 4 and len(digits) == 0:
            fields["company"] = text.replace(".", "").strip()
            break

    # If still empty, use company keyword rule
    if not fields["company"]:
        for line in line_texts[:20]:
            text = line["text"].strip()
            upper = text.upper()

            if len(text) < 4:
                continue

            if any(stop in upper for stop in stop_company_words):
                continue

            if any(keyword in upper for keyword in company_keywords):
                fields["company"] = text.replace(".", "").strip()
                break

    # Final fallback: first clean top line with mostly letters
    if not fields["company"]:
        for line in line_texts[:12]:
            text = line["text"].strip()
            upper = text.upper()

            if len(text) < 4:
                continue

            if any(stop in upper for stop in stop_company_words):
                continue

            letters = re.findall(r"[A-Za-z]", text)
            digits = re.findall(r"\d", text)

            if len(letters) >= 4 and len(digits) <= 1:
                fields["company"] = text.replace(".", "").strip()
                break

    # =====================================================
    # 6. ADDRESS extraction
    # =====================================================
    address_lines = []

    address_keywords = [
        "NO",
        "LOT",
        "PT",
        "JALAN",
        "ROAD",
        "STREET",
        "BANDAR",
        "TAMAN",
        "KLANG",
        "SHAH",
        "ALAM",
        "SELANGOR",
        "LANGOR",
        "JOHOR",
        "MASAI",
        "MALAYSIA",
        "NIGERIA",
        "CITY",
        "STATE"
    ]

    hard_stop_address_words = [
        "TEL",
        "TELEPHONE",
        "FAX",
        "EMAIL",
        "GST",
        "TAX INVOICE",
        "SIMPLIFIED",
        "CASH",
        "RECEIPT",
        "DATE",
        "TIME",
        "ITEM",
        "QTY",
        "PRICE",
        "AMOUNT",
        "TOTAL",
        "CASHIER",
        "PRE-AUTHORISATION",
        "PRE-AUTHORIZATION"
    ]

    skip_address_words = [
        "COMPANY NO",
        "ROC",
        "SITE",
        "GST NO"
    ]

    for line in line_texts:
        text = line["text"].strip()
        upper = text.upper()

        # Stop when receipt body starts after address already started
        if address_lines and any(stop in upper for stop in hard_stop_address_words):
            break

        # Skip company line
        if fields["company"] and text.strip().upper() == fields["company"].strip().upper():
            continue

        # Skip registration / phone / GST lines
        if any(skip in upper for skip in skip_address_words):
            continue

        # Skip obvious non-address body text
        if any(stop in upper for stop in hard_stop_address_words):
            continue

        # Check real address keywords safely
        if any(has_keyword(upper, keyword) for keyword in address_keywords):
            address_lines.append(text)

    if address_lines:
        address = " ".join(address_lines)
        address = re.sub(r"\s+", " ", address).strip()

        fields["address"] = address

    return fields

def predict_invoice(
    input_path: str,
    model_dir: str = "./layoutlm_invoice_model",
    output_json: str = "prediction_output.json",
    output_csv: str = "",
    lang: str = "eng",
    min_conf: float = 20.0,
    page: int = 1,
) -> Dict:
    model, tokenizer = load_model_and_tokenizer(model_dir)
    device = get_device()

    print(f"Using device: {device}")
    print("Running OCR...")
    ocr_result = run_tesseract_ocr(
        input_path=input_path,
        lang=lang,
        min_conf=min_conf,
        page=page,
    )

    words = ocr_result["words"]
    boxes = ocr_result["boxes"]

    if not words:
        raise ValueError(
            "No OCR words detected. Try lowering --min-conf or check the image quality."
        )

    print(f"OCR words detected: {len(words)}")
    print("Running LayoutLM prediction...")

    token_predictions = predict_word_labels(
        words=words,
        boxes=boxes,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    spans = build_spans(token_predictions)
    fields = reconstruct_fields(spans)

    if not fields.get("company") and not fields.get("date") and not fields.get("total"):
       print("LayoutLM returned empty fields. Using rule-based fallback...")
       fields = fallback_extract_from_tokens(token_predictions)

    verification = verify_extraction(fields)

    result = {
        "source": os.path.abspath(input_path),
        "model_dir": os.path.abspath(model_dir),
        "fields": fields,
        "verification": verification,
        "spans": spans,
        "tokens": token_predictions,
    }

    save_json(result, output_json)
    print(f"Saved JSON result to: {output_json}")

    if output_csv:
        save_csv(fields, verification, output_csv)
        print(f"Saved CSV result to: {output_csv}")

    print("\nExtracted Fields:")
    print(json.dumps(fields, indent=2, ensure_ascii=False))

    print("\nVerification:")
    print(json.dumps(verification, indent=2, ensure_ascii=False))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict invoice fields using trained LayoutLM model.")
    parser.add_argument("--input", required=True, help="Invoice image/PDF path")
    parser.add_argument("--model-dir", default="./layoutlm_invoice_model", help="Trained model directory")
    parser.add_argument("--output", default="prediction_output.json", help="Output JSON file")
    parser.add_argument("--csv", default="prediction_output.csv", help="Output CSV file. Use empty string to disable.")
    parser.add_argument("--lang", default="eng", help="Tesseract OCR language")
    parser.add_argument("--min-conf", type=float, default=20.0, help="Minimum OCR confidence")
    parser.add_argument("--page", type=int, default=1, help="PDF page number")

    args = parser.parse_args()

    predict_invoice(
        input_path=args.input,
        model_dir=args.model_dir,
        output_json=args.output,
        output_csv=args.csv,
        lang=args.lang,
        min_conf=args.min_conf,
        page=args.page,
    )


if __name__ == "__main__":
    main()
