"""Generate the two-page case-study PDF and render preview pages."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "case-study" / "merchant-revenue-forecasting.html"
PDF_PATH = ROOT / "case-study" / "Merchant_Revenue_Forecasting_Case_Study.pdf"
PREVIEW_DIR = ROOT / "case-study" / "_preview"
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
]


def find_chrome() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("Google Chrome is required to generate the case-study PDF.")


def print_pdf(chrome: Path) -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)


def set_metadata_and_render() -> None:
    doc = fitz.open(PDF_PATH)
    if doc.page_count != 2:
        raise RuntimeError(f"PDF has {doc.page_count} pages; expected exactly 2.")
    doc.set_metadata(
        {
            "title": "Merchant Revenue Forecasting",
            "author": "Efrain Castillo",
            "subject": "FP&A case study: default 12-month merchant revenue forecast method",
            "keywords": "FP&A, forecasting, merchant revenue, Holt, synthetic data, planning",
            "creator": "Merchant Revenue Forecasting case-study generator",
        }
    )
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for old in PREVIEW_DIR.glob("page_*.png"):
        old.unlink()
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pix.save(PREVIEW_DIR / f"page_{i}.png")
        text = page.get_text("text")
        if not text.strip():
            raise RuntimeError(f"PDF page {i} has no extractable text.")
    tmp = PDF_PATH.with_name("Merchant_Revenue_Forecasting_Case_Study.tmp.pdf")
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    tmp.replace(PDF_PATH)


def main() -> None:
    if not HTML_PATH.exists():
        raise FileNotFoundError(HTML_PATH)
    chrome = find_chrome()
    print_pdf(chrome)
    set_metadata_and_render()
    print(f"Wrote {PDF_PATH}")
    print(f"Rendered preview pages in {PREVIEW_DIR}")


if __name__ == "__main__":
    sys.exit(main())
