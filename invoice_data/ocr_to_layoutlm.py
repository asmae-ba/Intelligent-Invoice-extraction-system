"""
ocr_to_layoutlm.py

Converts a real invoice image/PDF into the format required by LayoutLM:
    words + normalized bounding boxes in the 0-1000 LayoutLM coordinate system.

Usage:
    python ocr_to_layoutlm.py --input samples/invoice.jpg --output processed/sample.json
    python ocr_to_layoutlm.py --input samples/invoice.pdf --output processed/sample.json

Required packages:
    pip install pillow pytesseract

For PDF support:
    pip install pdf2image
    Also install Poppler on your system.

For OCR support:
    Install Tesseract OCR on your system.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps, ImageFilter

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    from pytesseract import Output
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "pytesseract is required. Install it with: pip install pytesseract\n"
        "Also install Tesseract OCR on your computer."
    ) from exc


PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(input_path: str, page: int = 1, dpi: int = 220) -> Image.Image:
    """
    Loads an image file or converts a PDF page to a PIL image.

    Args:
        input_path: Path to invoice image or PDF.
        page: PDF page number, starting from 1.
        dpi: PDF conversion DPI.

    Returns:
        PIL Image in RGB format.
    """
    path = Path(input_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if suffix in PDF_EXTENSIONS:
        try:
            from pdf2image import convert_from_path
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PDF input requires pdf2image. Install it with: pip install pdf2image\n"
                "You may also need to install Poppler."
            ) from exc

        pages = convert_from_path(str(path), dpi=dpi, first_page=page, last_page=page)
        if not pages:
            raise ValueError(f"Could not read page {page} from PDF: {input_path}")
        return pages[0].convert("RGB")

    if suffix in IMAGE_EXTENSIONS:
        return Image.open(path).convert("RGB")

    raise ValueError(
        f"Unsupported file type: {suffix}. Use image files or PDF."
    )


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Light preprocessing before OCR.
    Keeps it simple and safe for invoices/receipts.
    """
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def normalize_bbox(box: Tuple[int, int, int, int], width: int, height: int) -> List[int]:
    """
    Converts pixel coordinates to LayoutLM 0-1000 normalized coordinates.

    Args:
        box: (x0, y0, x1, y1) in image pixels.
        width: image width.
        height: image height.

    Returns:
        [x0, y0, x1, y1] in LayoutLM scale.
    """
    x0, y0, x1, y1 = box

    def scale_x(x: int) -> int:
        return int(max(0, min(1000, round(1000 * x / width))))

    def scale_y(y: int) -> int:
        return int(max(0, min(1000, round(1000 * y / height))))

    return [scale_x(x0), scale_y(y0), scale_x(x1), scale_y(y1)]


def run_tesseract_ocr(
    input_path: str,
    lang: str = "eng",
    min_conf: float = 20.0,
    page: int = 1,
    dpi: int = 220,
) -> Dict:
    """
    Runs Tesseract OCR and returns words with bounding boxes.

    Returns:
        {
            "words": [...],
            "boxes": [[x0,y0,x1,y1], ...],
            "confidences": [...],
            "image_size": {"width": W, "height": H},
            "source": "..."
        }
    """
    image = load_image(input_path, page=page, dpi=dpi)
    width, height = image.size
    processed = preprocess_image(image)

    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        output_type=Output.DICT,
        config="--psm 6"
    )

    words: List[str] = []
    boxes: List[List[int]] = []
    confidences: List[float] = []

    total_items = len(data.get("text", []))

    for i in range(total_items):
        text = str(data["text"][i]).strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf < min_conf:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        if w <= 0 or h <= 0:
            continue

        box = normalize_bbox((x, y, x + w, y + h), width, height)

        words.append(text)
        boxes.append(box)
        confidences.append(conf)

    return {
        "words": words,
        "boxes": boxes,
        "confidences": confidences,
        "image_size": {"width": width, "height": height},
        "source": os.path.abspath(input_path),
        "page": page,
    }


def save_layoutlm_input(data: Dict, output_path: str) -> None:
    """Saves OCR result as JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR invoice and prepare LayoutLM input.")
    parser.add_argument("--input", required=True, help="Invoice image/PDF path")
    parser.add_argument("--output", default="layoutlm_input.json", help="Output JSON path")
    parser.add_argument("--lang", default="eng", help="Tesseract language, default: eng")
    parser.add_argument("--min-conf", type=float, default=20.0, help="Minimum OCR confidence")
    parser.add_argument("--page", type=int, default=1, help="PDF page number, starts from 1")
    parser.add_argument("--dpi", type=int, default=220, help="PDF conversion DPI")

    args = parser.parse_args()

    result = run_tesseract_ocr(
        input_path=args.input,
        lang=args.lang,
        min_conf=args.min_conf,
        page=args.page,
        dpi=args.dpi,
    )

    save_layoutlm_input(result, args.output)

    print("OCR complete.")
    print(f"Words detected: {len(result['words'])}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
