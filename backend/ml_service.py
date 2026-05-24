"""
ML valuation service for InvestIQ.

Purpose
-------
Classify one ticker as:
    - UNDERVALUED
    - FAIRLY_VALUED
    - OVERVALUED

Uses only:
    - data/sp500_complete.parquet through StaticDataLoader
    - models/valuation_model.joblib
    - models/scaler.joblib
    - models/model_metrics.json

Rules:
    - Does not train the model inside the API/backend.
    - Does not use yfinance.
    - Does not use SEC EDGAR.
    - Does not use Wikipedia.
    - Does not use live APIs.

If model artifacts are missing, the service falls back to deterministic
rule-based classification.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    from data_layer import StaticDataLoader
except ImportError:
    # Allows imports when used as backend.ml_service
    from backend.data_layer import StaticDataLoader


class MLValuationService:
    """
    ML-based and rule-based valuation classification service.

    The service loads the trained RandomForest model lazily. If model artifacts
    are missing, classify_ticker automatically uses rule_based_classify.
    """

    FEATURES = [
        "pe",
        "pb",
        "roe",
        "roa",
        "de",
        "current_ratio",
        "gross_margin",
    ]

    LABELS = [
        "UNDERVALUED",
        "FAIRLY_VALUED",
        "OVERVALUED",
    ]

    FEATURE_LABELS = {
        "pe": "P/E",
        "pb": "P/B",
        "roe": "ROE",
        "roa": "ROA",
        "de": "Debt-to-Equity",
        "current_ratio": "Current Ratio",
        "gross_margin": "Gross Margin",
    }

    def __init__(
        self,
        data_loader: StaticDataLoader | None = None,
        model_path: str = "models/valuation_model.joblib",
        scaler_path: str = "models/scaler.joblib",
        metrics_path: str = "models/model_metrics.json",
    ) -> None:
        """
        Create MLValuationService.

        Parameters
        ----------
        data_loader:
            Optional StaticDataLoader instance.
        model_path:
            Path to the trained valuation model.
        scaler_path:
            Path to the fitted StandardScaler.
        metrics_path:
            Path to model metrics JSON.
        """
        self.data_loader = data_loader or StaticDataLoader()

        self.model_path = self._resolve_path(model_path)
        self.scaler_path = self._resolve_path(scaler_path)
        self.metrics_path = self._resolve_path(metrics_path)

        self.model: Any | None = None
        self.scaler: Any | None = None
        self.metrics: dict[str, Any] = {}
        self.model_loaded = False
        self.model_available = False
        self.model_load_error: str | None = None

    @staticmethod
    def _project_root() -> Path:
        """
        Return project root from expected location:

            investiq/backend/ml_service.py
        """
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _resolve_path(cls, file_path: str) -> Path:
        """
        Resolve absolute or project-relative paths.
        """
        path = Path(file_path)

        if path.is_absolute():
            return path

        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        return cls._project_root() / path

    def load_model(self) -> None:
        """
        Load model, scaler, and metrics from local files.

        This method does not train anything. It only reads local artifacts.

        If any artifact is missing or unreadable, the service records the error
        and classify_ticker will use rule-based fallback.
        """
        if self.model_loaded:
            return

        self.model_loaded = True

        missing_files = [
            path for path in [self.model_path, self.scaler_path, self.metrics_path]
            if not path.exists()
        ]

        if missing_files:
            missing_text = ", ".join(str(path) for path in missing_files)
            self.model_available = False
            self.model_load_error = (
                "Model artifacts are missing. "
                f"Missing: {missing_text}. "
                "Run: python scripts/train_model.py"
            )
            return

        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

            with self.metrics_path.open("r", encoding="utf-8") as file:
                self.metrics = json.load(file)

            self.model_available = True
            self.model_load_error = None

        except Exception as exc:
            self.model_available = False
            self.model_load_error = (
                f"Could not load model artifacts: {exc}. "
                "Run: python scripts/train_model.py"
            )

    def classify_ticker(self, ticker: str) -> dict[str, Any]:
        """
        Classify a ticker using the trained model.

        If model files are unavailable, use rule-based fallback.

        Parameters
        ----------
        ticker:
            Company ticker. Matching is case-insensitive.

        Returns
        -------
        dict
            JSON-safe valuation classification payload.
        """
        normalized_ticker = self._normalize_ticker(ticker)
        self.load_model()

        if not self.model_available:
            fallback = self.rule_based_classify(normalized_ticker)
            fallback["model_error"] = self.model_load_error
            return self._json_safe(fallback)

        company = self.data_loader.get_company_by_ticker(normalized_ticker)
        feature_values = self._get_feature_values(normalized_ticker)

        X = pd.DataFrame([feature_values], columns=self.FEATURES)
        X_scaled = self.scaler.transform(X)

        prediction = str(self.model.predict(X_scaled)[0])

        probabilities = self._predict_probabilities(X_scaled)
        confidence = probabilities.get(prediction)

        top_positive_signals = self._build_positive_signals(feature_values)
        top_negative_signals = self._build_negative_signals(feature_values)

        payload = {
            "ticker": company.get("ticker", normalized_ticker),
            "company_name": company.get("company_name"),
            "verdict": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_accuracy": self._get_model_accuracy(),
            "classifier_type": "random_forest",
            "feature_importance": self._build_feature_importance(feature_values),
            "feature_values": self._format_feature_values(feature_values),
            "top_positive_signals": top_positive_signals,
            "top_negative_signals": top_negative_signals,
            "explanation": self._build_ml_explanation(
                company=company,
                verdict=prediction,
                confidence=confidence,
                feature_values=feature_values,
                positive_signals=top_positive_signals,
                negative_signals=top_negative_signals,
            ),
        }

        return self._json_safe(payload)

    def rule_based_classify(self, ticker: str) -> dict[str, Any]:
        """
        Classify a ticker using deterministic valuation rules.

        Fallback rules:

        UNDERVALUED:
            pe < 15 and roe > 0.15 and pb <= 3

        OVERVALUED:
            pe > 25 or roe < 0.08 or pb > 4

        FAIRLY_VALUED:
            otherwise
        """
        normalized_ticker = self._normalize_ticker(ticker)

        company = self.data_loader.get_company_by_ticker(normalized_ticker)
        feature_values = self._get_feature_values(normalized_ticker)

        pe = feature_values["pe"]
        pb = feature_values["pb"]
        roe = feature_values["roe"]

        if pe < 15 and roe > 0.15 and pb <= 3:
            verdict = "UNDERVALUED"
        elif pe > 25 or roe < 0.08 or pb > 4:
            verdict = "OVERVALUED"
        else:
            verdict = "FAIRLY_VALUED"

        probabilities = self._rule_based_probabilities(verdict, feature_values)
        confidence = probabilities[verdict]

        top_positive_signals = self._build_positive_signals(feature_values)
        top_negative_signals = self._build_negative_signals(feature_values)

        payload = {
            "ticker": company.get("ticker", normalized_ticker),
            "company_name": company.get("company_name"),
            "verdict": verdict,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_accuracy": self._get_model_accuracy(),
            "classifier_type": "rule_based_fallback",
            "feature_importance": self._build_feature_importance(feature_values),
            "feature_values": self._format_feature_values(feature_values),
            "top_positive_signals": top_positive_signals,
            "top_negative_signals": top_negative_signals,
            "explanation": self._build_rule_based_explanation(
                company=company,
                verdict=verdict,
                confidence=confidence,
                feature_values=feature_values,
                positive_signals=top_positive_signals,
                negative_signals=top_negative_signals,
            ),
        }

        return self._json_safe(payload)

    def _predict_probabilities(self, X_scaled: Any) -> dict[str, float]:
        """
        Return class probabilities in a stable label order.
        """
        if not hasattr(self.model, "predict_proba"):
            return {label: 0.0 for label in self.LABELS}

        probability_array = self.model.predict_proba(X_scaled)[0]
        model_classes = [str(label) for label in self.model.classes_]

        raw_probabilities = {
            label: float(probability)
            for label, probability in zip(model_classes, probability_array)
        }

        return {
            label: round(float(raw_probabilities.get(label, 0.0)), 4)
            for label in self.LABELS
        }

    def _get_feature_values(self, ticker: str) -> dict[str, float]:
        """
        Get normalized feature values for a ticker.
        """
        ratios = self.data_loader.get_ratios(ticker)

        feature_values: dict[str, float] = {}

        for feature in self.FEATURES:
            value = self._normalize_ratio_value(feature, ratios.get(feature))

            if value is None:
                raise ValueError(
                    f"Feature '{feature}' is missing for ticker '{ticker}'. "
                    "Rebuild the dataset with: python scripts/build_dataset.py"
                )

            feature_values[feature] = value

        return feature_values

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """
        Normalize ticker input.
        """
        if ticker is None or not str(ticker).strip():
            raise ValueError("Ticker must not be empty.")

        return str(ticker).strip().upper()

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        """
        Convert value to a finite float or None.
        """
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(number) or math.isinf(number):
            return None

        return number

    def _normalize_ratio_value(self, feature: str, value: Any) -> float | None:
        """
        Normalize Kaggle ratio scale inconsistencies.

        Examples:
            roe = 36 means 0.36
            gross_margin = 39 means 0.39
            current_ratio = 135 means 1.35
        """
        number = self._to_float_or_none(value)

        if number is None:
            return None

        if feature in {"roe", "gross_margin"} and abs(number) > 1.5:
            number = number / 100.0

        if feature == "current_ratio" and abs(number) > 20:
            number = number / 100.0

        return number

    def _get_model_accuracy(self) -> float | None:
        """
        Return stored model accuracy if available.
        """
        accuracy = self._to_float_or_none(self.metrics.get("accuracy"))
        if accuracy is None:
            return None
        return round(accuracy, 4)

    def _build_feature_importance(
        self,
        feature_values: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Return feature importance list sorted descending.

        If model metrics are missing, use a rule-based fallback order.
        """
        metric_importances = self.metrics.get("feature_importances", {})

        if not isinstance(metric_importances, dict) or not metric_importances:
            metric_importances = {
                "pe": 0.35,
                "pb": 0.20,
                "roe": 0.20,
                "roa": 0.10,
                "de": 0.05,
                "current_ratio": 0.05,
                "gross_margin": 0.05,
            }

        result = []

        for feature in self.FEATURES:
            importance = self._to_float_or_none(metric_importances.get(feature)) or 0.0
            value = feature_values.get(feature)

            result.append(
                {
                    "feature": feature,
                    "label": self.FEATURE_LABELS.get(feature, feature),
                    "importance": round(importance, 4),
                    "value": value,
                    "formatted_value": self._format_feature_value(feature, value),
                }
            )

        return sorted(result, key=lambda item: item["importance"], reverse=True)

    def _format_feature_values(
        self,
        feature_values: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        """
        Return feature values with labels and formatted values.
        """
        return {
            feature: {
                "label": self.FEATURE_LABELS.get(feature, feature),
                "value": value,
                "formatted_value": self._format_feature_value(feature, value),
            }
            for feature, value in feature_values.items()
        }

    @staticmethod
    def _format_feature_value(feature: str, value: float | None) -> str:
        """
        Format a ratio value for API/frontend display.
        """
        if value is None:
            return "N/A"

        if feature in {"pe", "pb"}:
            return f"{value:.2f}x"

        if feature in {"roe", "roa", "gross_margin"}:
            return f"{value * 100:.2f}%"

        return f"{value:.2f}"

    def _build_positive_signals(
        self,
        feature_values: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Generate positive valuation/quality signals from actual ratio values.
        """
        signals: list[dict[str, Any]] = []

        pe = feature_values["pe"]
        pb = feature_values["pb"]
        roe = feature_values["roe"]
        roa = feature_values["roa"]
        de = feature_values["de"]
        current_ratio = feature_values["current_ratio"]
        gross_margin = feature_values["gross_margin"]

        if pe < 15:
            signals.append(
                self._signal("pe", pe, "Low P/E compared with a basic value threshold.")
            )

        if pb <= 3:
            signals.append(
                self._signal("pb", pb, "Reasonable price compared with book value.")
            )

        if roe > 0.15:
            signals.append(
                self._signal("roe", roe, "Strong return on shareholder equity.")
            )

        if roa >= 0.05:
            signals.append(
                self._signal("roa", roa, "Good asset efficiency.")
            )

        if de <= 0.5:
            signals.append(
                self._signal("de", de, "Conservative debt level.")
            )

        if current_ratio >= 1.5:
            signals.append(
                self._signal("current_ratio", current_ratio, "Strong short-term liquidity.")
            )

        if gross_margin >= 0.40:
            signals.append(
                self._signal("gross_margin", gross_margin, "Strong gross margin.")
            )

        return signals[:5]

    def _build_negative_signals(
        self,
        feature_values: dict[str, float],
    ) -> list[dict[str, Any]]:
        """
        Generate negative valuation/quality signals from actual ratio values.
        """
        signals: list[dict[str, Any]] = []

        pe = feature_values["pe"]
        pb = feature_values["pb"]
        roe = feature_values["roe"]
        roa = feature_values["roa"]
        de = feature_values["de"]
        current_ratio = feature_values["current_ratio"]
        gross_margin = feature_values["gross_margin"]

        if pe > 25:
            signals.append(
                self._signal("pe", pe, "High P/E may indicate an expensive valuation.")
            )

        if pb > 4:
            signals.append(
                self._signal("pb", pb, "High P/B means the market price is high versus book value.")
            )

        if roe < 0.08:
            signals.append(
                self._signal("roe", roe, "Weak return on shareholder equity.")
            )

        if roa < 0.02:
            signals.append(
                self._signal("roa", roa, "Weak asset efficiency.")
            )

        if de > 1.5:
            signals.append(
                self._signal("de", de, "High debt compared with equity.")
            )

        if current_ratio < 1.0:
            signals.append(
                self._signal("current_ratio", current_ratio, "Weak short-term liquidity.")
            )

        if gross_margin < 0.20:
            signals.append(
                self._signal("gross_margin", gross_margin, "Low gross margin.")
            )

        return signals[:5]

    def _signal(
        self,
        feature: str,
        value: float,
        message: str,
    ) -> dict[str, Any]:
        """
        Build a signal object.
        """
        return {
            "feature": feature,
            "label": self.FEATURE_LABELS.get(feature, feature),
            "value": value,
            "formatted_value": self._format_feature_value(feature, value),
            "message": message,
        }

    def _rule_based_probabilities(
        self,
        verdict: str,
        feature_values: dict[str, float],
    ) -> dict[str, float]:
        """
        Generate simple deterministic probabilities for fallback mode.

        These are not model probabilities. They are confidence-style scores for
        the fallback explanation.
        """
        pe = feature_values["pe"]
        pb = feature_values["pb"]
        roe = feature_values["roe"]

        base = {
            "UNDERVALUED": 0.15,
            "FAIRLY_VALUED": 0.15,
            "OVERVALUED": 0.15,
        }

        if verdict == "UNDERVALUED":
            confidence = 0.75
            if pe < 12 and pb <= 2 and roe > 0.20:
                confidence = 0.85
            base["UNDERVALUED"] = confidence

        elif verdict == "OVERVALUED":
            confidence = 0.75
            if pe > 35 or pb > 6 or roe < 0.04:
                confidence = 0.85
            base["OVERVALUED"] = confidence

        else:
            confidence = 0.65
            if 15 <= pe <= 25 and 1.5 <= pb <= 4 and 0.08 <= roe <= 0.20:
                confidence = 0.75
            base["FAIRLY_VALUED"] = confidence

        remaining = 1.0 - base[verdict]
        other_labels = [label for label in self.LABELS if label != verdict]

        for label in other_labels:
            base[label] = remaining / 2.0

        return {
            label: round(base[label], 4)
            for label in self.LABELS
        }

    def _build_ml_explanation(
        self,
        company: dict[str, Any],
        verdict: str,
        confidence: float | None,
        feature_values: dict[str, float],
        positive_signals: list[dict[str, Any]],
        negative_signals: list[dict[str, Any]],
    ) -> str:
        """
        Build beginner-friendly explanation for ML output.
        """
        company_name = company.get("company_name") or company.get("ticker") or "The company"
        confidence_text = (
            f"{confidence * 100:.1f}%"
            if confidence is not None
            else "unknown"
        )

        core = (
            f"{company_name} is classified as {verdict} by the local Random Forest "
            f"model with {confidence_text} confidence. "
            f"The model used P/E {self._format_feature_value('pe', feature_values['pe'])}, "
            f"P/B {self._format_feature_value('pb', feature_values['pb'])}, "
            f"ROE {self._format_feature_value('roe', feature_values['roe'])}, "
            f"ROA {self._format_feature_value('roa', feature_values['roa'])}, "
            f"Debt-to-Equity {self._format_feature_value('de', feature_values['de'])}, "
            f"Current Ratio {self._format_feature_value('current_ratio', feature_values['current_ratio'])}, "
            f"and Gross Margin {self._format_feature_value('gross_margin', feature_values['gross_margin'])}."
        )

        signal_text = self._signal_summary(positive_signals, negative_signals)

        return f"{core} {signal_text}"

    def _build_rule_based_explanation(
        self,
        company: dict[str, Any],
        verdict: str,
        confidence: float,
        feature_values: dict[str, float],
        positive_signals: list[dict[str, Any]],
        negative_signals: list[dict[str, Any]],
    ) -> str:
        """
        Build beginner-friendly explanation for fallback output.
        """
        company_name = company.get("company_name") or company.get("ticker") or "The company"

        core = (
            f"{company_name} is classified as {verdict} using the rule-based fallback "
            f"because the trained model files were not available. "
            f"The fallback used P/E {self._format_feature_value('pe', feature_values['pe'])}, "
            f"P/B {self._format_feature_value('pb', feature_values['pb'])}, "
            f"and ROE {self._format_feature_value('roe', feature_values['roe'])}. "
            f"Fallback confidence is {confidence * 100:.1f}%."
        )

        signal_text = self._signal_summary(positive_signals, negative_signals)

        return f"{core} {signal_text}"

    @staticmethod
    def _signal_summary(
        positive_signals: list[dict[str, Any]],
        negative_signals: list[dict[str, Any]],
    ) -> str:
        """
        Summarize top signals in one short explanation sentence.
        """
        if positive_signals and negative_signals:
            return (
                f"Positive signals include {positive_signals[0]['label']} "
                f"({positive_signals[0]['formatted_value']}). "
                f"Negative signals include {negative_signals[0]['label']} "
                f"({negative_signals[0]['formatted_value']})."
            )

        if positive_signals:
            return (
                f"The strongest positive signal is {positive_signals[0]['label']} "
                f"({positive_signals[0]['formatted_value']})."
            )

        if negative_signals:
            return (
                f"The strongest negative signal is {negative_signals[0]['label']} "
                f"({negative_signals[0]['formatted_value']})."
            )

        return "No strong positive or negative ratio signal dominates the result."

    def _json_safe(self, value: Any) -> Any:
        """
        Convert pandas/numpy values into JSON-safe Python values.
        """
        if value is None:
            return None

        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._json_safe(item) for item in value]

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if hasattr(value, "item"):
            try:
                return self._json_safe(value.item())
            except Exception:
                pass

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        return value


def classify_ticker(ticker: str) -> dict[str, Any]:
    """
    Convenience function for API modules.
    """
    service = MLValuationService()
    return service.classify_ticker(ticker)


if __name__ == "__main__":
    service = MLValuationService()
    result = service.classify_ticker("AAPL")
    print(json.dumps(result, indent=2))
