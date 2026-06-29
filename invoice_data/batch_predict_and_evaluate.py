"""
batch_predict_and_evaluate.py

This script:
1. Runs predict_invoice.py on all invoice/receipt images in samples/
2. Saves each JSON result in outputs/
3. Creates evaluation/predictions_summary.csv
4. If evaluation/ground_truth.csv exists, compares predictions with ground truth
5. Creates evaluation/evaluation_report.csv and evaluation/evaluation_metrics.json

Run from the same folder where predict_invoice.py exists.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = str(value).upper().strip()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def text_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def normalize_amount(value: str):
    if value is None:
        return None

    value = str(value).strip()
    value = value.replace(",", ".")
    value = re.sub(r"[^0-9.]", "", value)

    if not value:
        return None

    try:
        return round(float(value), 2)
    except ValueError:
        return None


def normalize_date(value: str) -> str:
    """
    Normalize dates to YYYY-MM-DD when possible.

    Supports:
    15/01/2019
    23/03/17
    2026-06-26
    26 Mar 2018
    """
    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    value = re.sub(
        r"\s+\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?$",
        "",
        value,
        flags=re.IGNORECASE
    )

    formats = [
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%d %b %Y",
        "%d %B %Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    return normalize_text(value)


def load_json_result(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {
            "error": str(e),
            "fields": {},
            "verification": {
                "overall_status": "error"
            }
        }


def run_prediction(
    python_exe: str,
    predict_script: Path,
    image_path: Path,
    model_dir: Path,
    output_path: Path
) -> bool:
    command = [
        python_exe,
        str(predict_script),
        "--input",
        str(image_path),
        "--model-dir",
        str(model_dir),
        "--output",
        str(output_path),
    ]

    print("\nRunning:", image_path.name)

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True
        )

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            print(f"Failed: {image_path.name}")
            return False

        return True

    except Exception as e:
        print(f"Error running prediction for {image_path.name}: {e}")
        return False


def find_input_files(samples_dir: Path):
    files = []

    for path in samples_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)

    return sorted(files)


def save_prediction_summary(rows, summary_csv: Path):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "company",
                "address",
                "date",
                "total",
                "overall_status",
                "json_output"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)


def create_ground_truth_template(prediction_rows, ground_truth_path: Path):
    ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ground_truth_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "company",
                "address",
                "date",
                "total"
            ]
        )

        writer.writeheader()

        for row in prediction_rows:
            writer.writerow({
                "filename": row["filename"],
                "company": row["company"],
                "address": row["address"],
                "date": row["date"],
                "total": row["total"],
            })


def load_ground_truth(ground_truth_csv: Path):
    data = {}

    with open(ground_truth_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row.get("filename", "").strip()

            if filename:
                data[filename] = {
                    "company": row.get("company", ""),
                    "address": row.get("address", ""),
                    "date": row.get("date", ""),
                    "total": row.get("total", ""),
                }

    return data


def evaluate_predictions(
    prediction_rows,
    ground_truth_data,
    company_threshold=0.85,
    address_threshold=0.70
):
    report_rows = []

    totals = {
        "company_correct": 0,
        "address_correct": 0,
        "date_correct": 0,
        "total_correct": 0,
        "all_fields_correct": 0,
        "evaluated_files": 0,
    }

    for pred in prediction_rows:
        filename = pred["filename"]

        if filename not in ground_truth_data:
            continue

        truth = ground_truth_data[filename]

        company_score = text_similarity(pred["company"], truth["company"])
        address_score = text_similarity(pred["address"], truth["address"])

        pred_date = normalize_date(pred["date"])
        true_date = normalize_date(truth["date"])

        pred_total = normalize_amount(pred["total"])
        true_total = normalize_amount(truth["total"])

        company_correct = company_score >= company_threshold
        address_correct = address_score >= address_threshold
        date_correct = pred_date == true_date and pred_date != ""
        total_correct = pred_total == true_total and pred_total is not None

        all_correct = company_correct and address_correct and date_correct and total_correct

        totals["company_correct"] += int(company_correct)
        totals["address_correct"] += int(address_correct)
        totals["date_correct"] += int(date_correct)
        totals["total_correct"] += int(total_correct)
        totals["all_fields_correct"] += int(all_correct)
        totals["evaluated_files"] += 1

        report_rows.append({
            "filename": filename,

            "pred_company": pred["company"],
            "true_company": truth["company"],
            "company_similarity": round(company_score, 3),
            "company_correct": "Yes" if company_correct else "No",

            "pred_address": pred["address"],
            "true_address": truth["address"],
            "address_similarity": round(address_score, 3),
            "address_correct": "Yes" if address_correct else "No",

            "pred_date": pred["date"],
            "true_date": truth["date"],
            "normalized_pred_date": pred_date,
            "normalized_true_date": true_date,
            "date_correct": "Yes" if date_correct else "No",

            "pred_total": pred["total"],
            "true_total": truth["total"],
            "normalized_pred_total": pred_total,
            "normalized_true_total": true_total,
            "total_correct": "Yes" if total_correct else "No",

            "all_fields_correct": "Yes" if all_correct else "No",
            "overall_status": pred["overall_status"],
        })

    return report_rows, totals


def save_evaluation_report(report_rows, report_csv: Path):
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    if not report_rows:
        return

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)


def save_metrics(totals, metrics_json: Path):
    metrics_json.parent.mkdir(parents=True, exist_ok=True)

    n = totals["evaluated_files"]

    if n == 0:
        metrics = {
            "message": "No files were evaluated. Add evaluation/ground_truth.csv first."
        }
    else:
        metrics = {
            "evaluated_files": n,
            "company_accuracy": round(totals["company_correct"] / n, 4),
            "address_accuracy": round(totals["address_correct"] / n, 4),
            "date_accuracy": round(totals["date_correct"] / n, 4),
            "total_accuracy": round(totals["total_correct"] / n, 4),
            "all_fields_exact_accuracy": round(totals["all_fields_correct"] / n, 4),
        }

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Batch run invoice prediction and evaluate results."
    )

    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--evaluation-dir", default="evaluation")
    parser.add_argument("--model-dir", default="layoutlm_invoice_model")
    parser.add_argument("--predict-script", default="predict_invoice.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-predict", action="store_true")

    args = parser.parse_args()

    base_dir = Path.cwd()

    samples_dir = base_dir / args.samples_dir
    outputs_dir = base_dir / args.outputs_dir
    evaluation_dir = base_dir / args.evaluation_dir
    model_dir = base_dir / args.model_dir
    predict_script = base_dir / args.predict_script

    outputs_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if not samples_dir.exists():
        print(f"Samples folder not found: {samples_dir}")
        return

    if not model_dir.exists():
        print(f"Model folder not found: {model_dir}")
        return

    if not predict_script.exists():
        print(f"Prediction script not found: {predict_script}")
        return

    input_files = find_input_files(samples_dir)

    if not input_files:
        print(f"No image/PDF files found in: {samples_dir}")
        return

    print(f"Found {len(input_files)} files.")

    prediction_rows = []

    for image_path in input_files:
        output_path = outputs_dir / f"{image_path.stem}_result.json"

        if not args.skip_predict:
            success = run_prediction(
                python_exe=args.python,
                predict_script=predict_script,
                image_path=image_path,
                model_dir=model_dir,
                output_path=output_path
            )

            if not success:
                continue

        result = load_json_result(output_path)

        fields = result.get("fields", {})
        verification = result.get("verification", {})

        prediction_rows.append({
            "filename": image_path.name,
            "company": fields.get("company", ""),
            "address": fields.get("address", ""),
            "date": fields.get("date", ""),
            "total": fields.get("total", ""),
            "overall_status": verification.get("overall_status", ""),
            "json_output": str(output_path),
        })

    summary_csv = evaluation_dir / "predictions_summary.csv"
    save_prediction_summary(prediction_rows, summary_csv)

    print(f"\nPrediction summary saved to: {summary_csv}")

    ground_truth_csv = evaluation_dir / "ground_truth.csv"

    if not ground_truth_csv.exists():
        template_csv = evaluation_dir / "ground_truth_template.csv"
        create_ground_truth_template(prediction_rows, template_csv)

        print("\nGround truth file not found.")
        print(f"A template was created here: {template_csv}")
        print("Open it, correct the values manually, then rename/copy it to ground_truth.csv.")
        print("After that, run this script again with:")
        print("python batch_predict_and_evaluate.py --skip-predict")
        return

    ground_truth_data = load_ground_truth(ground_truth_csv)

    report_rows, totals = evaluate_predictions(
        prediction_rows=prediction_rows,
        ground_truth_data=ground_truth_data
    )

    report_csv = evaluation_dir / "evaluation_report.csv"
    metrics_json = evaluation_dir / "evaluation_metrics.json"

    save_evaluation_report(report_rows, report_csv)
    metrics = save_metrics(totals, metrics_json)

    print(f"Evaluation report saved to: {report_csv}")
    print(f"Evaluation metrics saved to: {metrics_json}")

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
