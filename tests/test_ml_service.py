"""
Tests for backend/ml_service.py.
"""

from __future__ import annotations

from backend.ml_service import MLValuationService


def test_rule_based_fallback_works_if_model_files_are_missing(data_loader, tmp_path):
    missing_model = tmp_path / "models" / "missing_model.joblib"
    missing_scaler = tmp_path / "models" / "missing_scaler.joblib"
    missing_metrics = tmp_path / "models" / "missing_metrics.json"

    service = MLValuationService(
        data_loader=data_loader,
        model_path=str(missing_model),
        scaler_path=str(missing_scaler),
        metrics_path=str(missing_metrics),
    )

    result = service.classify_ticker("AAA")

    assert result["ticker"] == "AAA"
    assert result["classifier_type"] == "rule_based_fallback"
    assert result["verdict"] in {"UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"}
    assert 0 <= result["confidence"] <= 1
    assert "model_error" in result


def test_classification_output_has_verdict_confidence_and_explanation(data_loader, tmp_path):
    service = MLValuationService(
        data_loader=data_loader,
        model_path=str(tmp_path / "no_model.joblib"),
        scaler_path=str(tmp_path / "no_scaler.joblib"),
        metrics_path=str(tmp_path / "no_metrics.json"),
    )

    result = service.classify_ticker("CCC")

    assert "verdict" in result
    assert "confidence" in result
    assert "explanation" in result
    assert isinstance(result["explanation"], str)
    assert result["explanation"]
    assert result["verdict"] in {"UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"}
    assert 0 <= result["confidence"] <= 1
