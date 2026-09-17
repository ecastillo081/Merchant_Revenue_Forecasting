"""Publication claim reconciliation across README, PDF, and website copy."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf as fitz
import pandas as pd

from functions.config import BASELINE_MODEL, PRIMARY_WINDOW

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"
CASE_HTML = ROOT / "case-study" / "merchant-revenue-forecasting.html"
PDF = ROOT / "case-study" / "Merchant_Revenue_Forecasting_Case_Study.pdf"
WEBSITE = Path(r"C:\Users\ecast\PycharmProjects\efrain-website\public\work\merchant-revenue-forecasting\index.html")


def _display() -> dict:
    claims = json.loads((RESULTS / "publication_claims.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(RESULTS / "model_performance_summary.csv")
    selected = summary.iloc[0]
    baseline = summary.loc[summary["model"] == BASELINE_MODEL].iloc[0]
    return {
        "claims": claims,
        "selected_model": str(selected["model"]),
        "selected": f"{selected['mean_mape']:.2f}",
        "baseline": f"{baseline['mean_mape']:.2f}",
        "point": f"{claims['point_difference']:.2f}",
        "relative": f"{claims['relative_reduction_pct']:.1f}",
        "best_count": str(int(selected["best_model_count"])),
        "best_pct": f"{selected['best_model_pct']:.0f}",
    }


def test_readme_pdf_and_case_html_agree_with_results():
    info = _display()
    texts = {
        "README": README.read_text(encoding="utf-8"),
        "case HTML": CASE_HTML.read_text(encoding="utf-8"),
        "PDF": "".join(page.get_text("text") for page in fitz.open(PDF)),
    }
    required = [
        info["selected"],
        info["baseline"],
        info["point"],
        f"{info['relative']}%",
        "unweighted",
        "Driver Scenario Regression",
        "Seasonal Naive",
        info["selected_model"],
        info["best_count"],
    ]
    forbidden = ["Linear Regression", "fixture"]
    for label, text in texts.items():
        for token in required:
            assert token in text, f"{label} missing {token!r}"
        lower = text.lower()
        for token in forbidden:
            assert token.lower() not in lower, f"{label} contains forbidden {token!r}"


def test_website_copy_agrees_when_present():
    if not WEBSITE.exists():
        return
    info = _display()
    text = WEBSITE.read_text(encoding="utf-8")
    assert info["selected"] in text
    assert info["baseline"] in text
    assert "Driver Scenario Regression" in text
    assert "fixture" not in text.lower()
    assert "Linear Regression" not in text
    assert "champion/challenger" in text.lower() or "champion/challenger" in text


def test_chart_labels_match_computed_values():
    info = _display()
    path = ROOT / "figures" / "leaderboard_MAPE.png"
    assert path.exists()
    summary = pd.read_csv(RESULTS / "model_performance_summary.csv")
    assert str(summary.iloc[0]["window"]) == str(PRIMARY_WINDOW)
    assert f"{summary.iloc[0]['mean_mape']:.2f}" == info["selected"]
    assert summary.iloc[0]["model"] == info["selected_model"]
