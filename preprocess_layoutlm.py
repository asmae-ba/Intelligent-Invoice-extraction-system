"""
Convert organized SROIE annotations into LayoutLM-style JSONL files.

Input:
    invoice_data/raw/sroie/{train,test}/images
    invoice_data/raw/sroie/{train,test}/annotations

Output:
    invoice_data/processed/sroie/{train,test}.jsonl
    invoice_data/processed/sroie/labels.txt
    invoice_data/processed/sroie/summary.json
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).parent.absolute()
RAW_DIR = SCRIPT_DIR / "invoice_data" / "raw" / "sroie"
OUTPUT_DIR = SCRIPT_DIR / "invoice_data" / "processed" / "sroie"

FIELD_TO_LABEL = {
    "address": "ADDRESS",
    "company": "COMPANY",
    "date": "DATE",
    "total": "TOTAL",
}

LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]


def read_text(path):
    for encoding in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


def normalize_text(value):
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokenise(value):
    return re.findall(r"\S+", value)


def parse_annotation(path):
    content = read_text(path)
    entity_block = content.split("ENTITIES (Ground Truth)", 1)[1].split("OCR BOXES", 1)[0]
    entity_json = entity_block.replace("=" * 60, "").strip()
    entities = json.loads(entity_json)

    ocr_block = content.split("OCR BOXES", 1)[1].replace("=" * 60, "").strip()
    lines = []
    for raw_line in ocr_block.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        parts = raw_line.split(",", 8)
        if len(parts) != 9:
            continue

        try:
            coords = [int(part) for part in parts[:8]]
        except ValueError:
            continue

        xs = coords[0::2]
        ys = coords[1::2]
        text = parts[8].strip()
        if text:
            lines.append(
                {
                    "text": text,
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                    "norm": normalize_text(text),
                }
            )

    return entities, lines


def normalise_box(box, width, height):
    x0, y0, x1, y1 = box
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]

    return [
        max(0, min(1000, round(1000 * x0 / width))),
        max(0, min(1000, round(1000 * y0 / height))),
        max(0, min(1000, round(1000 * x1 / width))),
        max(0, min(1000, round(1000 * y1 / height))),
    ]


def similarity(a, b):
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def pick_label_for_line(line, entities):
    norm = line["norm"]
    if not norm:
        return "O"

    best_label = "O"
    best_score = 0.0

    for field, label in FIELD_TO_LABEL.items():
        value = str(entities.get(field, "")).strip()
        target = normalize_text(value)
        if not target:
            continue

        if field in ("date", "total"):
            score = 1.0 if target and target in norm else 0.0
        elif field == "address":
            # Address often spans multiple OCR lines. Label partial address lines too.
            score = 1.0 if norm in target or target in norm else similarity(norm, target)
        else:
            score = similarity(norm, target)

        if score > best_score:
            best_score = score
            best_label = label

    thresholds = {
        "COMPANY": 0.72,
        "DATE": 1.0,
        "ADDRESS": 0.55,
        "TOTAL": 1.0,
    }

    if best_label != "O" and best_score >= thresholds[best_label]:
        return best_label
    return "O"


def build_example(annotation_path, image_path):
    entities, lines = parse_annotation(annotation_path)

    with Image.open(image_path) as image:
        width, height = image.size

    words = []
    boxes = []
    ner_tags = []
    raw_line_labels = []

    for line in lines:
        label = pick_label_for_line(line, entities)
        raw_line_labels.append(label)
        line_words = tokenise(line["text"])
        line_box = normalise_box(line["box"], width, height)

        for index, word in enumerate(line_words):
            words.append(word)
            boxes.append(line_box)
            if label == "O":
                ner_tags.append("O")
            else:
                prefix = "B" if index == 0 else "I"
                ner_tags.append(f"{prefix}-{label}")

    return {
        "id": annotation_path.stem,
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "entities": {
            "company": entities.get("company"),
            "date": entities.get("date"),
            "address": entities.get("address"),
            "total": entities.get("total"),
        },
        "words": words,
        "boxes": boxes,
        "ner_tags": ner_tags,
        "line_label_counts": {
            label: raw_line_labels.count(label)
            for label in ["COMPANY", "DATE", "ADDRESS", "TOTAL", "O"]
        },
    }


def image_for_annotation(images_dir, annotation_path):
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{annotation_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def process_split(split):
    annotations_dir = RAW_DIR / split / "annotations"
    images_dir = RAW_DIR / split / "images"
    output_path = OUTPUT_DIR / f"{split}.jsonl"

    counts = {
        "documents": 0,
        "missing_images": 0,
        "errors": 0,
        "tokens": 0,
        "labeled_tokens": 0,
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        for annotation_path in sorted(annotations_dir.glob("*.txt")):
            image_path = image_for_annotation(images_dir, annotation_path)
            if image_path is None:
                counts["missing_images"] += 1
                continue

            try:
                example = build_example(annotation_path, image_path)
            except Exception as exc:
                counts["errors"] += 1
                print(f"Warning: failed to process {annotation_path.name}: {exc}")
                continue

            counts["documents"] += 1
            counts["tokens"] += len(example["words"])
            counts["labeled_tokens"] += sum(tag != "O" for tag in example["ner_tags"])
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    return counts


def main():
    print("=" * 60)
    print("SROIE LayoutLM preprocessing")
    print("=" * 60)
    print(f"Raw data: {RAW_DIR}")
    print(f"Output:   {OUTPUT_DIR}")

    if not RAW_DIR.exists():
        raise SystemExit(f"Raw data folder not found: {RAW_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")

    summary = {}
    for split in ("train", "test"):
        print(f"\nProcessing {split}...")
        summary[split] = process_split(split)
        counts = summary[split]
        print(f"  Documents:      {counts['documents']}")
        print(f"  Tokens:         {counts['tokens']}")
        print(f"  Labeled tokens: {counts['labeled_tokens']}")
        print(f"  Missing images: {counts['missing_images']}")
        print(f"  Errors:         {counts['errors']}")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nDone. Processed files:")
    print(f"  {OUTPUT_DIR / 'train.jsonl'}")
    print(f"  {OUTPUT_DIR / 'test.jsonl'}")
    print(f"  {OUTPUT_DIR / 'labels.txt'}")
    print(f"  {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
