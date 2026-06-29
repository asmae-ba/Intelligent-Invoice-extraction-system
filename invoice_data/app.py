"""
app.py - Streamlit Web Interface for Invoice Extraction Project

Run:
    streamlit run app.py

Required project structure:
    invoice_data/
    ├── app.py
    ├── predict_invoice.py
    ├── ocr_to_layoutlm.py
    ├── verify_extraction.py
    ├── layoutlm_invoice_model/
    ├── uploads/
    └── outputs/

Install:
    pip install streamlit pandas pillow
"""

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Invoice Extraction System",
    page_icon="🧾",
    layout="wide"
)


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "layoutlm_invoice_model"
PREDICT_SCRIPT = BASE_DIR / "predict_invoice.py"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded invoice image/PDF to uploads folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    file_path = UPLOAD_DIR / f"{timestamp}_{safe_name}"

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def run_prediction(input_path: Path) -> tuple[bool, Path, str]:
    """Run predict_invoice.py and return success status, output path, and logs."""
    output_path = OUTPUT_DIR / f"{input_path.stem}_result.json"

    command = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--input",
        str(input_path),
        "--model-dir",
        str(MODEL_DIR),
        "--output",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False
    )

    logs = ""
    if result.stdout:
        logs += result.stdout

    if result.stderr:
        logs += "\n" + result.stderr

    success = result.returncode == 0 and output_path.exists()

    return success, output_path, logs


def load_json(path: Path) -> dict:
    """Load JSON output."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_reviewed_json(original_result: dict, reviewed_fields: dict, save_path: Path) -> Path:
    """Save reviewed/corrected result."""
    reviewed_result = dict(original_result)
    reviewed_result["reviewed_fields"] = reviewed_fields
    reviewed_result["final_fields"] = reviewed_fields
    reviewed_result["review_status"] = "human_reviewed"
    reviewed_result["reviewed_at"] = datetime.now().isoformat(timespec="seconds")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(reviewed_result, f, indent=2, ensure_ascii=False)

    return save_path


def convert_fields_to_csv(fields: dict) -> str:
    """Convert final fields to CSV string."""
    fieldnames = ["company", "address", "date", "total"]
    output_lines = []
    output_lines.append(",".join(fieldnames))

    row = []
    for field in fieldnames:
        value = str(fields.get(field, "")).replace('"', '""')
        row.append(f'"{value}"')

    output_lines.append(",".join(row))

    return "\n".join(output_lines)


def show_image_preview(file_path: Path):
    """Display uploaded image preview when possible."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

    if file_path.suffix.lower() in image_extensions:
        image = Image.open(file_path)
        st.image(image, caption="Uploaded Invoice / Receipt", use_container_width=True)
    else:
        st.info("PDF uploaded. Preview is not shown here, but extraction can still run.")


def show_verification(result: dict):
    """Display verification checks."""
    verification = result.get("verification", {})
    checks = verification.get("checks", {})
    status = verification.get("overall_status", "unknown")

    if status == "valid":
        st.success(f"Verification Status: {status}")
    elif status == "needs_review":
        st.warning(f"Verification Status: {status}")
    else:
        st.info(f"Verification Status: {status}")

    if checks:
        rows = []

        for field_name, check in checks.items():
            rows.append({
                "Field": field_name,
                "Status": check.get("status", ""),
                "Message": check.get("message", ""),
                "Normalized": check.get("normalized", "")
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🧾 Invoice IDP")
st.sidebar.write("Automated Invoice Data Extraction and Verification")

st.sidebar.markdown("---")
st.sidebar.write("**System Flow**")
st.sidebar.write("Upload → OCR → LayoutLM → Fallback → Verification → Review → Export")
st.sidebar.markdown("---")

if MODEL_DIR.exists():
    st.sidebar.success("Model folder found")
else:
    st.sidebar.error("Model folder missing")

if PREDICT_SCRIPT.exists():
    st.sidebar.success("predict_invoice.py found")
else:
    st.sidebar.error("predict_invoice.py missing")


# =====================================================
# MAIN PAGE
# =====================================================

st.title("🧾 Intelligent Invoice Extraction System")
st.write(
    "Upload an invoice or receipt image/PDF. The system extracts company, address, date, and total amount, "
    "then verifies the result and allows human correction before export."
)

tab1, tab2, tab3 = st.tabs([
    "1. Upload & Extract",
    "2. Review & Correct",
    "3. Export Result"
])


# =====================================================
# TAB 1: UPLOAD & EXTRACT
# =====================================================

with tab1:
    st.header("Upload Invoice / Receipt")

    uploaded_file = st.file_uploader(
        "Choose an invoice image or PDF",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "pdf"]
    )

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)

        st.session_state["uploaded_file_path"] = str(file_path)
        st.success(f"File uploaded: {file_path.name}")

        col1, col2 = st.columns([1, 1])

        with col1:
            show_image_preview(file_path)

        with col2:
            st.subheader("Extraction")

            if st.button("Run Extraction", type="primary"):
                if not MODEL_DIR.exists():
                    st.error("Model folder not found: layoutlm_invoice_model")
                elif not PREDICT_SCRIPT.exists():
                    st.error("predict_invoice.py not found")
                else:
                    with st.spinner("Running OCR + LayoutLM + fallback extraction..."):
                        success, output_path, logs = run_prediction(file_path)

                    st.session_state["logs"] = logs

                    if success:
                        result = load_json(output_path)

                        st.session_state["result_path"] = str(output_path)
                        st.session_state["result"] = result
                        st.session_state["fields"] = result.get("fields", {})

                        st.success("Extraction completed successfully.")
                    else:
                        st.error("Extraction failed. Check logs below.")

            if "logs" in st.session_state:
                with st.expander("Show system logs"):
                    st.code(st.session_state["logs"])

    if "result" in st.session_state:
        st.markdown("---")
        st.subheader("Raw Extracted Fields")

        fields = st.session_state["result"].get("fields", {})
        st.json(fields)

        show_verification(st.session_state["result"])


# =====================================================
# TAB 2: REVIEW & CORRECT
# =====================================================

with tab2:
    st.header("Human Review and Correction")

    if "result" not in st.session_state:
        st.info("Please upload and extract an invoice first.")
    else:
        result = st.session_state["result"]
        fields = result.get("fields", {})
        uploaded_path = Path(st.session_state.get("uploaded_file_path", ""))

        col1, col2 = st.columns([1, 1])

        with col1:
            if uploaded_path.exists():
                show_image_preview(uploaded_path)

        with col2:
            st.subheader("Edit Extracted Fields")

            company = st.text_input("Company / Vendor Name", value=fields.get("company", ""))
            address = st.text_area("Address", value=fields.get("address", ""), height=100)
            date = st.text_input("Date", value=fields.get("date", ""))
            total = st.text_input("Total Amount", value=fields.get("total", ""))

            reviewed_fields = {
                "company": company.strip(),
                "address": address.strip(),
                "date": date.strip(),
                "total": total.strip()
            }

            st.session_state["reviewed_fields"] = reviewed_fields

            if st.button("Save Reviewed Result", type="primary"):
                result_path = Path(st.session_state["result_path"])
                reviewed_path = OUTPUT_DIR / f"{result_path.stem}_reviewed.json"

                save_reviewed_json(
                    original_result=result,
                    reviewed_fields=reviewed_fields,
                    save_path=reviewed_path
                )

                st.session_state["reviewed_path"] = str(reviewed_path)
                st.success(f"Reviewed result saved: {reviewed_path.name}")

        st.markdown("---")
        st.subheader("Verification Details")
        show_verification(result)


# =====================================================
# TAB 3: EXPORT RESULT
# =====================================================

with tab3:
    st.header("Export Final Result")

    if "result" not in st.session_state:
        st.info("Please upload and extract an invoice first.")
    else:
        final_fields = st.session_state.get(
            "reviewed_fields",
            st.session_state["result"].get("fields", {})
        )

        st.subheader("Final Fields")
        st.json(final_fields)

        json_data = json.dumps(final_fields, indent=2, ensure_ascii=False)
        csv_data = convert_fields_to_csv(final_fields)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="invoice_extraction_result.json",
                mime="application/json"
            )

        with col2:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="invoice_extraction_result.csv",
                mime="text/csv"
            )

        if "reviewed_path" in st.session_state:
            st.success(f"Reviewed JSON saved at: {st.session_state['reviewed_path']}")

        if "result_path" in st.session_state:
            st.info(f"Original prediction JSON saved at: {st.session_state['result_path']}")
