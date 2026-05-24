"""
Peer comparison service for InvestIQ.

Purpose
-------
Compare a selected company with similar companies using only the local
Kaggle-derived Parquet dataset through StaticDataLoader.

No yfinance.
No SEC EDGAR.
No Wikipedia.
No Kaggle API at runtime.
No live network calls.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

try:
    from data_layer import StaticDataLoader
except ImportError:
    from backend.data_layer import StaticDataLoader


class PeerComparisonService:
    """
    Build peer-comparison payloads for financial-dashboard views.

    Raw target and peer rows are preserved for display.

    Averages, bar chart data, and radar chart data use cleaned metric values so
    negative or extreme valuation values do not distort the dashboard.
    """

    METRIC_KEYS = [
        "pe",
        "pb",
        "roe",
        "roa",
        "de",
        "current_ratio",
        "gross_margin",
    ]

    HIGHER_IS_BETTER = {
        "roe",
        "roa",
        "current_ratio",
        "gross_margin",
    }

    LOWER_IS_BETTER = {
        "pe",
        "pb",
        "de",
    }

    METRIC_LABELS = {
        "pe": "P/E",
        "pb": "P/B",
        "roe": "ROE",
        "roa": "ROA",
        "de": "Debt/Equity",
        "current_ratio": "Current Ratio",
        "gross_margin": "Gross Margin",
    }

    OUTPUT_COLUMNS = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "market_cap",
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

    def get_peer_comparison_payload(
        self,
        ticker: str,
        limit: int = 8,
    ) -> dict[str, Any]:
        """
        Return a full peer-comparison payload for one ticker.
        """
        normalized_ticker = self._normalize_ticker(ticker)
        safe_limit = self._normalize_limit(limit)

        companies = self.data_loader.load_all_companies().copy()
        companies = self._prepare_company_frame(companies)

        target_row = self._find_target_row(companies, normalized_ticker)
        peers = self._select_peers(companies, target_row, safe_limit)

        target_df = pd.DataFrame([target_row])
        comparison_df = pd.concat([target_df, peers], ignore_index=True)

        sector_df = self._filter_same_value(
            companies,
            column="sector",
            value=target_row.get("sector"),
        )
        industry_df = self._filter_same_value(
            companies,
            column="industry",
            value=target_row.get("industry"),
        )

        target_record = self._row_to_record(target_row)
        peer_records = self._dataframe_to_records(peers)

        sector_averages = self._calculate_averages(sector_df)
        industry_averages = self._calculate_averages(industry_df)
        peer_averages = self._calculate_averages(peers)

        cleaned_sector_df = self._clean_statistics_frame(sector_df)
        cleaned_comparison_df = self._clean_statistics_frame(comparison_df)

        normalization_universe = (
            cleaned_sector_df
            if not cleaned_sector_df.empty
            else cleaned_comparison_df
        )

        return self._json_safe(
            {
                "target": target_record,
                "peers": peer_records,
                "sector_averages": sector_averages,
                "industry_averages": industry_averages,
                "bar_chart_data": self._build_bar_chart_data(
                    target=target_record,
                    peer_averages=peer_averages,
                    sector_averages=sector_averages,
                    industry_averages=industry_averages,
                ),
                "radar_chart_data": self._build_radar_chart_data(
                    target=target_record,
                    peer_averages=peer_averages,
                    sector_averages=sector_averages,
                    industry_averages=industry_averages,
                    normalization_universe=normalization_universe,
                ),
            }
        )

    def _prepare_company_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare company data for comparison.
        """
        prepared = df.copy()

        for column in self.OUTPUT_COLUMNS:
            if column not in prepared.columns:
                prepared[column] = pd.NA

        prepared["ticker"] = prepared["ticker"].astype("string").str.strip().str.upper()

        for text_col in ["company_name", "sector", "industry"]:
            prepared[text_col] = prepared[text_col].astype("string").str.strip()

        if "year" in prepared.columns:
            prepared["year"] = pd.to_numeric(prepared["year"], errors="coerce")
            prepared = prepared.sort_values(["ticker", "year"], na_position="first")
            prepared = prepared.drop_duplicates(subset=["ticker"], keep="last")
        else:
            prepared = prepared.drop_duplicates(subset=["ticker"], keep="last")

        for metric in ["market_cap", *self.METRIC_KEYS]:
            prepared[metric] = pd.to_numeric(prepared[metric], errors="coerce")

        for metric in self.METRIC_KEYS:
            prepared[metric] = prepared[metric].apply(
                lambda value, key=metric: self._normalize_ratio_value(key, value)
            )

        return prepared.reset_index(drop=True)

    def _find_target_row(
        self,
        companies: pd.DataFrame,
        ticker: str,
    ) -> dict[str, Any]:
        match = companies[companies["ticker"] == ticker]

        if match.empty:
            raise KeyError(
                f"Ticker '{ticker}' was not found in the local InvestIQ dataset."
            )

        return match.iloc[0].to_dict()

    def _select_peers(
        self,
        companies: pd.DataFrame,
        target_row: dict[str, Any],
        limit: int,
    ) -> pd.DataFrame:
        """
        Select peers using same industry first, then same sector if needed.
        """
        target_ticker = target_row.get("ticker")
        target_industry = target_row.get("industry")
        target_sector = target_row.get("sector")

        non_target = companies[companies["ticker"] != target_ticker].copy()

        industry_peers = self._filter_same_value(
            non_target,
            column="industry",
            value=target_industry,
        )

        selected = industry_peers.copy()

        if len(selected) < limit:
            sector_peers = self._filter_same_value(
                non_target,
                column="sector",
                value=target_sector,
            )

            already_selected = set(selected["ticker"].dropna().tolist())
            sector_peers = sector_peers[
                ~sector_peers["ticker"].isin(already_selected)
            ]

            selected = pd.concat([selected, sector_peers], ignore_index=True)

        selected = selected.drop_duplicates(subset=["ticker"], keep="first")

        if "market_cap" in selected.columns:
            selected = selected.sort_values(
                by="market_cap",
                ascending=False,
                na_position="last",
            )

        return selected.head(limit)[self.OUTPUT_COLUMNS].reset_index(drop=True)

    @staticmethod
    def _filter_same_value(
        df: pd.DataFrame,
        column: str,
        value: Any,
    ) -> pd.DataFrame:
        """
        Filter a DataFrame by one value.

        Missing sector or industry returns an empty DataFrame.
        """
        if column not in df.columns:
            return df.iloc[0:0].copy()

        if value is None or pd.isna(value) or str(value).strip() == "":
            return df.iloc[0:0].copy()

        return df[df[column].astype("string").str.strip() == str(value).strip()].copy()

    def _calculate_averages(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate averages after invalid values are removed.
        """
        if df.empty:
            return {metric: None for metric in self.METRIC_KEYS}

        cleaned = self._clean_statistics_frame(df)
        averages: dict[str, Any] = {}

        for metric in self.METRIC_KEYS:
            if metric not in cleaned.columns:
                averages[metric] = None
                continue

            value = pd.to_numeric(cleaned[metric], errors="coerce").mean(skipna=True)
            averages[metric] = self._to_float_or_none(value)

        return self._json_safe(averages)

    def _clean_statistics_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace invalid metric values with NA for statistics and charts.
        """
        cleaned = df.copy()

        for metric in self.METRIC_KEYS:
            if metric not in cleaned.columns:
                cleaned[metric] = pd.NA
                continue

            cleaned[metric] = cleaned[metric].apply(
                lambda value, key=metric: self._clean_metric_for_statistics(key, value)
            )

        return cleaned

    def _clean_metric_for_statistics(self, metric: str, value: Any) -> float | None:
        """
        Remove values that should not be used in averages or chart normalization.

        Display rows still keep their raw values.
        """
        number = self._to_float_or_none(value)
        if number is None:
            return None

        if metric == "pe":
            if number <= 0 or number > 200:
                return None

        elif metric == "pb":
            if number <= 0 or number > 100:
                return None

        elif metric == "de":
            if number < 0 or number > 50:
                return None

        elif metric == "current_ratio":
            if number < 0 or number > 25:
                return None

        elif metric in {"roe", "roa"}:
            if number < -2 or number > 2:
                return None

        elif metric == "gross_margin":
            if number < -1 or number > 1:
                return None

        return number

    def _build_bar_chart_data(
        self,
        target: dict[str, Any],
        peer_averages: dict[str, Any],
        sector_averages: dict[str, Any],
        industry_averages: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build chart-ready metric values for grouped bar charts.
        """
        labels = [self.METRIC_LABELS[metric] for metric in self.METRIC_KEYS]

        return self._json_safe(
            {
                "metric_keys": self.METRIC_KEYS,
                "labels": labels,
                "target": [
                    self._clean_metric_for_statistics(metric, target.get(metric))
                    for metric in self.METRIC_KEYS
                ],
                "peer_average": [
                    peer_averages.get(metric) for metric in self.METRIC_KEYS
                ],
                "sector_average": [
                    sector_averages.get(metric) for metric in self.METRIC_KEYS
                ],
                "industry_average": [
                    industry_averages.get(metric) for metric in self.METRIC_KEYS
                ],
                "average_method": "mean_after_invalid_values_removed",
                "note": (
                    "Bar chart values use cleaned financial metrics. "
                    "Invalid values such as negative P/E, negative P/B, and "
                    "negative Debt/Equity are excluded from averages."
                ),
            }
        )

    def _build_radar_chart_data(
        self,
        target: dict[str, Any],
        peer_averages: dict[str, Any],
        sector_averages: dict[str, Any],
        industry_averages: dict[str, Any],
        normalization_universe: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Build chart-ready 0-100 scores for a future polar/radar chart.
        """
        labels = [self.METRIC_LABELS[metric] for metric in self.METRIC_KEYS]

        return self._json_safe(
            {
                "metric_keys": self.METRIC_KEYS,
                "labels": labels,
                "target": [
                    self._normalize_metric_to_score(
                        metric,
                        target.get(metric),
                        normalization_universe,
                    )
                    for metric in self.METRIC_KEYS
                ],
                "peer_average": [
                    self._normalize_metric_to_score(
                        metric,
                        peer_averages.get(metric),
                        normalization_universe,
                    )
                    for metric in self.METRIC_KEYS
                ],
                "sector_average": [
                    self._normalize_metric_to_score(
                        metric,
                        sector_averages.get(metric),
                        normalization_universe,
                    )
                    for metric in self.METRIC_KEYS
                ],
                "industry_average": [
                    self._normalize_metric_to_score(
                        metric,
                        industry_averages.get(metric),
                        normalization_universe,
                    )
                    for metric in self.METRIC_KEYS
                ],
                "scale": {
                    "min": 0,
                    "max": 100,
                },
                "higher_is_better": sorted(self.HIGHER_IS_BETTER),
                "lower_is_better": sorted(self.LOWER_IS_BETTER),
                "note": (
                    "Radar values are normalized 0-100 scores. "
                    "A higher score is always better, including for lower-is-better "
                    "metrics such as P/E, P/B, and Debt/Equity."
                ),
            }
        )

    def _normalize_metric_to_score(
        self,
        metric: str,
        value: Any,
        universe: pd.DataFrame,
    ) -> float | None:
        """
        Normalize one metric value to a 0-100 score.
        """
        number = self._clean_metric_for_statistics(metric, value)
        if number is None:
            return None

        if metric not in universe.columns:
            return None

        series = pd.to_numeric(universe[metric], errors="coerce").dropna()

        if series.empty:
            return None

        min_value = float(series.min())
        max_value = float(series.max())

        if math.isclose(min_value, max_value):
            return 50.0

        if metric in self.LOWER_IS_BETTER:
            score = (max_value - number) / (max_value - min_value) * 100.0
        else:
            score = (number - min_value) / (max_value - min_value) * 100.0

        score = max(0.0, min(100.0, score))
        return round(score, 2)

    def _row_to_record(self, row: dict[str, Any]) -> dict[str, Any]:
        record = {column: row.get(column) for column in self.OUTPUT_COLUMNS}
        return self._json_safe(record)

    def _dataframe_to_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []

        safe_df = df.copy()

        for column in self.OUTPUT_COLUMNS:
            if column not in safe_df.columns:
                safe_df[column] = pd.NA

        safe_df = safe_df[self.OUTPUT_COLUMNS]
        return [self._json_safe(row) for row in safe_df.to_dict(orient="records")]

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        if not ticker or not str(ticker).strip():
            raise ValueError("Ticker must not be empty.")
        return str(ticker).strip().upper()

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        try:
            safe_limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer.") from exc

        if safe_limit < 1:
            raise ValueError("limit must be at least 1.")

        return safe_limit

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(number) or math.isinf(number):
            return None

        return number

    def _normalize_ratio_value(self, metric: str, value: Any) -> float | None:
        """
        Normalize known Kaggle ratio scale inconsistencies.

        Examples:
            roe = 36 means 0.36
            gross_margin = 39 means 0.39
            current_ratio = 135 means 1.35
        """
        number = self._to_float_or_none(value)
        if number is None:
            return None

        if metric in {"roe", "gross_margin"} and abs(number) > 1.5:
            return number / 100.0

        if metric == "current_ratio" and abs(number) > 20:
            return number / 100.0

        return number

    def _json_safe(self, value: Any) -> Any:
        """
        Convert pandas/numpy values into JSON-safe Python values.
        """
        if value is None:
            return None

        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._json_safe(item) for item in value]

        if pd.isna(value):
            return None

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


def get_peer_comparison_payload(ticker: str, limit: int = 8) -> dict[str, Any]:
    """
    Convenience function for API modules.
    """
    service = PeerComparisonService()
    return service.get_peer_comparison_payload(ticker=ticker, limit=limit)


if __name__ == "__main__":
    service = PeerComparisonService()
    result = service.get_peer_comparison_payload("AAPL", limit=8)
    print(json.dumps(result, indent=2))
