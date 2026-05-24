"""
Ratio analysis service for InvestIQ.

This module uses only local Parquet-backed data through StaticDataLoader.
It does not call yfinance, SEC EDGAR, Wikipedia, Kaggle APIs, or any live
network service.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

try:
    from data_layer import StaticDataLoader
except ImportError:
    from backend.data_layer import StaticDataLoader


RatioStatus = Literal["green", "yellow", "red", "grey"]


class RatioService:
    """
    Create beginner-friendly financial ratio analysis for a company ticker.
    """

    RATIO_KEYS = [
        "pe",
        "pb",
        "roe",
        "roa",
        "de",
        "current_ratio",
        "gross_margin",
    ]

    def __init__(self, data_loader: StaticDataLoader | None = None) -> None:
        self.data_loader = data_loader or StaticDataLoader()

    def analyze_ticker(self, ticker: str) -> dict[str, Any]:
        """
        Return company metadata, ratio cards, financial health score, and summary.
        """
        normalized_ticker = self._normalize_ticker(ticker)
        company = self.data_loader.get_company_by_ticker(normalized_ticker)
        raw_ratios = self.data_loader.get_ratios(normalized_ticker)

        ratio_cards = [
            self._build_ratio_card(key, raw_ratios.get(key), raw_ratios)
            for key in self.RATIO_KEYS
        ]

        score = self._calculate_financial_health_score(ratio_cards)

        return self._json_safe(
            {
                "ticker": company.get("ticker", normalized_ticker),
                "company_name": company.get("company_name"),
                "sector": company.get("sector"),
                "ratios": ratio_cards,
                "financial_health_score": score,
                "summary": self._build_summary(company, ratio_cards, score),
            }
        )

    def get_analysis(self, ticker: str) -> dict[str, Any]:
        """
        Alias used by API code.
        """
        return self.analyze_ticker(ticker)

    def _build_ratio_card(
        self,
        key: str,
        raw_value: Any,
        raw_ratios: dict[str, Any],
    ) -> dict[str, Any]:
        value = self._normalize_ratio_value(key, raw_value)
        status = self._status_for_ratio(key, value)

        return self._json_safe(
            {
                "key": key,
                "label": self._label_for_ratio(key),
                "value": value,
                "formatted_value": self._format_ratio_value(key, value),
                "status": status,
                "explanation": self._explanation_for_ratio(key, status),
                "good_range": self._good_range_for_ratio(key),
                "bad_range": self._bad_range_for_ratio(key),
                "sector_percentile": self._extract_sector_percentile(key, raw_ratios),
            }
        )

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        if not ticker or not str(ticker).strip():
            raise ValueError("Ticker must not be empty.")
        return str(ticker).strip().upper()

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(number) or math.isinf(number):
            return None

        return number

    def _normalize_ratio_value(self, key: str, value: Any) -> float | None:
        """
        Normalize Kaggle ratio scale inconsistencies.

        The dgawlik/nyse fundamentals file may store some ratios as whole
        numbers instead of decimals:
        - ROE: 36 means 36%, not 3600%
        - Gross Margin: 39 means 39%, not 3900%
        - Current Ratio: 135 means 1.35, not 135
        """
        number = self._to_float(value)
        if number is None:
            return None

        if key in {"roe", "gross_margin"} and abs(number) > 1.5:
            number = number / 100.0

        if key == "current_ratio" and abs(number) > 20:
            number = number / 100.0

        return number

    def _extract_sector_percentile(
        self,
        key: str,
        raw_ratios: dict[str, Any],
    ) -> float | None:
        direct_key = f"{key}_sector_percentile"

        percentile = self._to_float(raw_ratios.get(direct_key))
        if percentile is not None:
            return round(percentile, 2)

        sector_percentiles = raw_ratios.get("sector_percentiles")
        if isinstance(sector_percentiles, dict):
            percentile = self._to_float(sector_percentiles.get(key))
            if percentile is not None:
                return round(percentile, 2)

        nested_value = raw_ratios.get(key)
        if isinstance(nested_value, dict):
            percentile = self._to_float(nested_value.get("sector_percentile"))
            if percentile is not None:
                return round(percentile, 2)

        return None

    def _status_for_ratio(self, key: str, value: float | None) -> RatioStatus:
        if value is None:
            return "grey"

        if key == "pe":
            if value <= 15:
                return "green"
            if value <= 25:
                return "yellow"
            return "red"

        if key == "pb":
            if value <= 1.5:
                return "green"
            if value <= 3:
                return "yellow"
            return "red"

        if key == "roe":
            if value >= 0.15:
                return "green"
            if value >= 0.08:
                return "yellow"
            return "red"

        if key == "roa":
            if value >= 0.05:
                return "green"
            if value >= 0.02:
                return "yellow"
            return "red"

        if key == "de":
            if value <= 0.5:
                return "green"
            if value <= 1.5:
                return "yellow"
            return "red"

        if key == "current_ratio":
            if value >= 1.5:
                return "green"
            if value >= 1.0:
                return "yellow"
            return "red"

        if key == "gross_margin":
            if value >= 0.40:
                return "green"
            if value >= 0.20:
                return "yellow"
            return "red"

        return "grey"

    @staticmethod
    def _score_for_status(status: RatioStatus) -> int:
        if status == "green":
            return 100
        if status == "yellow":
            return 60
        if status == "red":
            return 25
        return 0

    def _calculate_financial_health_score(
        self,
        ratio_cards: list[dict[str, Any]],
    ) -> int:
        if not ratio_cards:
            return 0

        scores = [
            self._score_for_status(str(card.get("status", "grey")))
            for card in ratio_cards
        ]
        return int(round(sum(scores) / len(scores)))

    @staticmethod
    def _label_for_ratio(key: str) -> str:
        labels = {
            "pe": "Price-to-Earnings",
            "pb": "Price-to-Book",
            "roe": "Return on Equity",
            "roa": "Return on Assets",
            "de": "Debt-to-Equity",
            "current_ratio": "Current Ratio",
            "gross_margin": "Gross Margin",
        }
        return labels.get(key, key)

    @staticmethod
    def _good_range_for_ratio(key: str) -> str:
        ranges = {
            "pe": "<= 15",
            "pb": "<= 1.5",
            "roe": ">= 15%",
            "roa": ">= 5%",
            "de": "<= 0.5",
            "current_ratio": ">= 1.5",
            "gross_margin": ">= 40%",
        }
        return ranges.get(key, "N/A")

    @staticmethod
    def _bad_range_for_ratio(key: str) -> str:
        ranges = {
            "pe": "> 25",
            "pb": "> 3",
            "roe": "< 8%",
            "roa": "< 2%",
            "de": "> 1.5",
            "current_ratio": "< 1.0",
            "gross_margin": "< 20%",
        }
        return ranges.get(key, "N/A")

    @staticmethod
    def _format_ratio_value(key: str, value: float | None) -> str:
        if value is None:
            return "N/A"

        if key in {"pe", "pb"}:
            return f"{value:.2f}x"

        if key in {"roe", "roa", "gross_margin"}:
            return f"{value * 100:.2f}%"

        if key in {"de", "current_ratio"}:
            return f"{value:.2f}"

        return f"{value:.2f}"

    @staticmethod
    def _explanation_for_ratio(key: str, status: RatioStatus) -> str:
        explanations = {
            "pe": {
                "green": "The stock looks reasonably priced compared with its earnings.",
                "yellow": "The stock has a moderate earnings valuation.",
                "red": "The stock looks expensive compared with its earnings.",
                "grey": "The P/E ratio is unavailable for this company.",
            },
            "pb": {
                "green": "The stock price is low compared with the company book value.",
                "yellow": "The stock price is moderately valued compared with book value.",
                "red": "The stock price is high compared with the company book value.",
                "grey": "The P/B ratio is unavailable for this company.",
            },
            "roe": {
                "green": "The company generates strong profit from shareholder equity.",
                "yellow": "The company generates acceptable profit from shareholder equity.",
                "red": "The company generates weak profit from shareholder equity.",
                "grey": "ROE is unavailable for this company.",
            },
            "roa": {
                "green": "The company uses its assets efficiently to generate profit.",
                "yellow": "The company has acceptable asset efficiency.",
                "red": "The company has weak profit generation compared with its assets.",
                "grey": "ROA is unavailable for this company.",
            },
            "de": {
                "green": "The company has a conservative debt level compared with equity.",
                "yellow": "The company has a moderate debt level compared with equity.",
                "red": "The company has a high debt level compared with equity.",
                "grey": "Debt-to-equity is unavailable for this company.",
            },
            "current_ratio": {
                "green": "The company has a strong short-term liquidity position.",
                "yellow": "The company has an acceptable short-term liquidity position.",
                "red": "The company may have weak short-term liquidity.",
                "grey": "Current ratio is unavailable for this company.",
            },
            "gross_margin": {
                "green": "The company keeps a strong share of revenue after direct costs.",
                "yellow": "The company keeps a moderate share of revenue after direct costs.",
                "red": "The company keeps a weak share of revenue after direct costs.",
                "grey": "Gross margin is unavailable for this company.",
            },
        }
        return explanations.get(key, {}).get(status, "No explanation available.")

    @staticmethod
    def _build_summary(
        company: dict[str, Any],
        ratio_cards: list[dict[str, Any]],
        score: int,
    ) -> str:
        company_name = company.get("company_name") or company.get("ticker") or "The company"

        green_count = sum(1 for ratio in ratio_cards if ratio.get("status") == "green")
        yellow_count = sum(1 for ratio in ratio_cards if ratio.get("status") == "yellow")
        red_count = sum(1 for ratio in ratio_cards if ratio.get("status") == "red")
        grey_count = sum(1 for ratio in ratio_cards if ratio.get("status") == "grey")

        if score >= 75:
            profile = "a strong financial profile"
        elif score >= 50:
            profile = "a mixed financial profile"
        else:
            profile = "a weak financial profile"

        summary = (
            f"{company_name} shows {profile} based on the latest available local dataset. "
            f"Ratio status count: {green_count} green, {yellow_count} yellow, {red_count} red"
        )

        if grey_count:
            summary += f", {grey_count} unavailable"

        return summary + "."

    def _json_safe(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._json_safe(item) for item in value]

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


if __name__ == "__main__":
    service = RatioService()
    result = service.analyze_ticker("AAPL")
    print(json.dumps(result, indent=2))
