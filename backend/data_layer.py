"""
Static local data access layer for InvestIQ.

This module reads only local Parquet files created by:

    python scripts/build_dataset.py

Runtime rules:
    - No yfinance
    - No SEC EDGAR
    - No Wikipedia
    - No Kaggle API
    - No live network calls
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import math

import pandas as pd


class StaticDataLoader:
    """
    Offline-first data loader for InvestIQ.

    The loader reads local Parquet files lazily, validates the required schema
    on first load, and keeps loaded DataFrames cached in memory for reuse by
    FastAPI services.
    """

    FUNDAMENTALS_REQUIRED_COLUMNS = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "year",
        "period_ending",
        "revenue",
        "gross_profit",
        "net_income",
        "total_assets",
        "total_equity",
        "total_debt",
        "current_assets",
        "current_liabilities",
        "eps",
        "estimated_shares",
        "year_end_price",
        "market_cap",
        "pe",
        "pb",
        "roe",
        "roa",
        "de",
        "current_ratio",
        "gross_margin",
    ]

    PRICES_REQUIRED_COLUMNS = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    LATEST_COMPANY_COLUMNS = [
        "ticker",
        "company_name",
        "sector",
        "industry",
        "year",
        "revenue",
        "net_income",
        "total_assets",
        "total_equity",
        "year_end_price",
        "market_cap",
        "pe",
        "pb",
        "roe",
        "roa",
        "de",
        "current_ratio",
        "gross_margin",
    ]

    HISTORICAL_FINANCIAL_COLUMNS = [
        "year",
        "revenue",
        "net_income",
        "total_assets",
        "total_equity",
        "gross_profit",
        "total_debt",
    ]

    RATIO_COLUMNS = [
        "pe",
        "pb",
        "roe",
        "roa",
        "de",
        "current_ratio",
        "gross_margin",
    ]

    PEER_COMPARISON_COLUMNS = [
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

    PRICE_HISTORY_COLUMNS = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    def __init__(
        self,
        fundamentals_path: str = "data/sp500_complete.parquet",
        prices_path: str = "data/sp500_prices.parquet",
    ) -> None:
        """
        Create a StaticDataLoader.

        Args:
            fundamentals_path:
                Path to the clean company fundamentals Parquet file.
            prices_path:
                Path to the clean price history Parquet file.
        """
        self.fundamentals_path = self._resolve_path(fundamentals_path)
        self.prices_path = self._resolve_path(prices_path)

        self._fundamentals_df: pd.DataFrame | None = None
        self._prices_df: pd.DataFrame | None = None
        self._latest_companies_df: pd.DataFrame | None = None

    @staticmethod
    def _project_root() -> Path:
        """
        Return the project root based on this file location.

        Expected location:
            investiq/backend/data_layer.py

        Project root:
            investiq/
        """
        return Path(__file__).resolve().parents[1]

    @classmethod
    def _resolve_path(cls, file_path: str) -> Path:
        """
        Resolve a file path.

        Absolute paths are used directly.
        Relative paths are first checked from the current working directory,
        then from the project root.
        """
        path = Path(file_path)

        if path.is_absolute():
            return path

        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        return cls._project_root() / path

    @staticmethod
    def _normalise_ticker(ticker: str) -> str:
        """
        Normalize ticker input for case-insensitive matching.
        """
        if ticker is None:
            raise KeyError("Ticker is required.")

        normalized = str(ticker).strip().upper()

        if not normalized:
            raise KeyError("Ticker is required.")

        return normalized

    @staticmethod
    def _json_safe_value(value: Any) -> Any:
        """
        Convert pandas/numpy values into JSON-safe Python values.
        """
        if pd.isna(value):
            return None

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value

        if isinstance(value, (str, int, bool)) or value is None:
            return value

        return value

    @classmethod
    def _json_safe_record(cls, record: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a flat dictionary into JSON-safe values.
        """
        return {
            str(key): cls._json_safe_value(value)
            for key, value in record.items()
        }

    @staticmethod
    def _empty_dataframe(columns: list[str]) -> pd.DataFrame:
        """
        Return an empty DataFrame with the requested columns.
        """
        return pd.DataFrame(columns=columns)

    @staticmethod
    def _validate_columns(
        df: pd.DataFrame,
        required_columns: list[str],
        source_path: Path,
    ) -> None:
        """
        Validate that a DataFrame contains the required columns.

        Raises:
            ValueError:
                If one or more required columns are missing.
        """
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            available_columns = "\n".join(f"  - {col}" for col in df.columns)
            missing_columns = ", ".join(missing)

            raise ValueError(
                f"Dataset schema is invalid: {source_path}\n"
                f"Missing required columns: {missing_columns}\n"
                f"Available columns:\n{available_columns}\n"
                "Rebuild the dataset with:\n"
                "  python scripts/build_dataset.py"
            )

    def _load_fundamentals(self) -> pd.DataFrame:
        """
        Lazily load and validate the fundamentals Parquet file.
        """
        if self._fundamentals_df is not None:
            return self._fundamentals_df

        if not self.fundamentals_path.exists():
            raise FileNotFoundError(
                f"Missing fundamentals Parquet file: {self.fundamentals_path}\n"
                "Create it by running:\n"
                "  python scripts/build_dataset.py"
            )

        df = pd.read_parquet(self.fundamentals_path)
        self._validate_columns(
            df=df,
            required_columns=self.FUNDAMENTALS_REQUIRED_COLUMNS,
            source_path=self.fundamentals_path,
        )

        df = df.copy()
        df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
        df["sector"] = df["sector"].astype("string").str.strip()
        df["industry"] = df["industry"].astype("string").str.strip()
        df["company_name"] = df["company_name"].astype("string").str.strip()
        df["period_ending"] = pd.to_datetime(df["period_ending"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        self._fundamentals_df = df
        return self._fundamentals_df

    def _load_prices(self) -> pd.DataFrame:
        """
        Lazily load and validate the prices Parquet file.
        """
        if self._prices_df is not None:
            return self._prices_df

        if not self.prices_path.exists():
            raise FileNotFoundError(
                f"Missing prices Parquet file: {self.prices_path}\n"
                "Create it by running:\n"
                "  python scripts/build_dataset.py"
            )

        df = pd.read_parquet(self.prices_path)
        self._validate_columns(
            df=df,
            required_columns=self.PRICES_REQUIRED_COLUMNS,
            source_path=self.prices_path,
        )

        df = df.copy()
        df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        self._prices_df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return self._prices_df

    def load_all_companies(self) -> pd.DataFrame:
        """
        Return one latest company record per ticker.

        Returns:
            pandas.DataFrame:
                Latest company records with financial and valuation fields.
        """
        if self._latest_companies_df is not None:
            return self._latest_companies_df.copy()

        df = self._load_fundamentals()

        latest = (
            df.sort_values(["ticker", "year", "period_ending"], na_position="first")
            .drop_duplicates(subset=["ticker"], keep="last")
            .sort_values("ticker")
            .reset_index(drop=True)
        )

        latest = latest[self.LATEST_COMPANY_COLUMNS].copy()
        self._latest_companies_df = latest

        return latest.copy()

    def get_company_by_ticker(self, ticker: str) -> dict[str, Any]:
        """
        Return the latest company record for a ticker.

        Args:
            ticker:
                Stock ticker. Matching is case-insensitive.

        Raises:
            KeyError:
                If the ticker does not exist in the dataset.
        """
        return dict(self._get_company_by_ticker_cached(self._normalise_ticker(ticker)))

    @lru_cache(maxsize=1024)
    def _get_company_by_ticker_cached(self, normalized_ticker: str) -> dict[str, Any]:
        """
        Cached implementation of latest ticker lookup.
        """
        companies = self.load_all_companies()
        result = companies[companies["ticker"] == normalized_ticker]

        if result.empty:
            raise KeyError(
                f"Ticker '{normalized_ticker}' was not found in the local InvestIQ dataset."
            )

        record = result.iloc[0].to_dict()
        return self._json_safe_record(record)

    def get_companies_by_sector(self, sector: str) -> pd.DataFrame:
        """
        Return latest company records for a sector.

        Args:
            sector:
                Sector name. Matching is case-insensitive.

        Returns:
            pandas.DataFrame:
                Empty DataFrame if sector is missing or not found.
        """
        if sector is None or not str(sector).strip():
            return self._empty_dataframe(self.LATEST_COMPANY_COLUMNS)

        requested_sector = str(sector).strip().lower()
        companies = self.load_all_companies()

        result = companies[
            companies["sector"].astype("string").str.strip().str.lower()
            == requested_sector
        ]

        if result.empty:
            return self._empty_dataframe(self.LATEST_COMPANY_COLUMNS)

        return result.sort_values(["market_cap", "ticker"], ascending=[False, True]).reset_index(
            drop=True
        )

    def get_historical_financials(self, ticker: str, years: int = 5) -> pd.DataFrame:
        """
        Return annual historical financials for a ticker.

        Args:
            ticker:
                Stock ticker. Matching is case-insensitive.
            years:
                Maximum number of latest annual rows to return.

        Raises:
            KeyError:
                If the ticker does not exist in the dataset.
        """
        normalized_ticker = self._normalise_ticker(ticker)
        years = max(int(years), 1)

        df = self._load_fundamentals()
        result = df[df["ticker"] == normalized_ticker]

        if result.empty:
            raise KeyError(
                f"Ticker '{normalized_ticker}' was not found in the local InvestIQ dataset."
            )

        result = (
            result.sort_values("year", ascending=False)
            .head(years)
            .sort_values("year")
            .reset_index(drop=True)
        )

        return result[self.HISTORICAL_FINANCIAL_COLUMNS].copy()

    def get_ratios(self, ticker: str) -> dict[str, Any]:
        """
        Return latest valuation and financial ratios for a ticker.

        The result also includes the ticker's percentile within its sector for
        each ratio. Percentiles are raw numeric percentiles, where a higher
        ratio value means a higher percentile.

        Args:
            ticker:
                Stock ticker. Matching is case-insensitive.

        Raises:
            KeyError:
                If the ticker does not exist in the dataset.
        """
        return dict(self._get_ratios_cached(self._normalise_ticker(ticker)))

    @lru_cache(maxsize=1024)
    def _get_ratios_cached(self, normalized_ticker: str) -> dict[str, Any]:
        """
        Cached implementation of ratio lookup.
        """
        companies = self.load_all_companies()
        target = companies[companies["ticker"] == normalized_ticker]

        if target.empty:
            raise KeyError(
                f"Ticker '{normalized_ticker}' was not found in the local InvestIQ dataset."
            )

        target_row = target.iloc[0]
        sector = target_row.get("sector")

        result: dict[str, Any] = {
            "ticker": normalized_ticker,
            "sector": sector,
        }

        if pd.isna(sector):
            sector_df = self._empty_dataframe(self.LATEST_COMPANY_COLUMNS)
        else:
            sector_df = companies[
                companies["sector"].astype("string").str.strip().str.lower()
                == str(sector).strip().lower()
            ]

        for ratio in self.RATIO_COLUMNS:
            value = target_row.get(ratio)
            result[ratio] = value

            percentile_key = f"{ratio}_sector_percentile"
            result[percentile_key] = None

            if not sector_df.empty and ratio in sector_df.columns and pd.notna(value):
                ratio_series = pd.to_numeric(sector_df[ratio], errors="coerce")
                valid_sector_values = ratio_series.dropna()

                if not valid_sector_values.empty:
                    percentile = (
                        valid_sector_values.rank(method="average", pct=True)
                        .loc[target_row.name]
                        if target_row.name in valid_sector_values.index
                        else None
                    )

                    if percentile is not None and pd.notna(percentile):
                        result[percentile_key] = float(percentile * 100)

        return self._json_safe_record(result)

    def get_peer_comparison(self, ticker: str, limit: int = 8) -> pd.DataFrame:
        """
        Return target company plus peer companies.

        Peer selection:
            1. Prefer same industry.
            2. If there are not enough peers, add companies from the same sector.
            3. Within each group, prefer closest market capitalization.

        Args:
            ticker:
                Stock ticker. Matching is case-insensitive.
            limit:
                Maximum number of peer companies to return, excluding the target.

        Raises:
            KeyError:
                If the ticker does not exist in the dataset.
        """
        normalized_ticker = self._normalise_ticker(ticker)
        limit = max(int(limit), 1)

        companies = self.load_all_companies()
        target = companies[companies["ticker"] == normalized_ticker]

        if target.empty:
            raise KeyError(
                f"Ticker '{normalized_ticker}' was not found in the local InvestIQ dataset."
            )

        target_row = target.iloc[0]
        target_sector = target_row.get("sector")
        target_industry = target_row.get("industry")
        target_market_cap = target_row.get("market_cap")

        peer_sets: list[pd.DataFrame] = []

        if pd.notna(target_industry):
            same_industry = companies[
                (companies["ticker"] != normalized_ticker)
                & (
                    companies["industry"].astype("string").str.strip().str.lower()
                    == str(target_industry).strip().lower()
                )
            ].copy()

            if not same_industry.empty:
                same_industry["_peer_priority"] = 1
                peer_sets.append(same_industry)

        if pd.notna(target_sector):
            same_sector = companies[
                (companies["ticker"] != normalized_ticker)
                & (
                    companies["sector"].astype("string").str.strip().str.lower()
                    == str(target_sector).strip().lower()
                )
            ].copy()

            if not same_sector.empty:
                same_sector["_peer_priority"] = 2
                peer_sets.append(same_sector)

        if peer_sets:
            peers = pd.concat(peer_sets, ignore_index=True)
            peers = peers.drop_duplicates(subset=["ticker"], keep="first")

            if pd.notna(target_market_cap):
                peers["_market_cap_distance"] = (
                    pd.to_numeric(peers["market_cap"], errors="coerce")
                    - float(target_market_cap)
                ).abs()
            else:
                peers["_market_cap_distance"] = pd.NA

            peers = (
                peers.sort_values(
                    ["_peer_priority", "_market_cap_distance", "market_cap", "ticker"],
                    ascending=[True, True, False, True],
                    na_position="last",
                )
                .head(limit)
                .copy()
            )
        else:
            peers = self._empty_dataframe(
                self.PEER_COMPARISON_COLUMNS
                + ["_peer_priority", "_market_cap_distance"]
            )

        target_for_output = target.copy()
        target_for_output["_peer_priority"] = 0
        target_for_output["_market_cap_distance"] = 0

        output = pd.concat([target_for_output, peers], ignore_index=True)
        output = output[self.PEER_COMPARISON_COLUMNS].reset_index(drop=True)

        return output.copy()

    def get_price_history(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Return daily price history for a ticker.

        Args:
            ticker:
                Stock ticker. Matching is case-insensitive.
            start_date:
                Optional inclusive start date in YYYY-MM-DD format.
            end_date:
                Optional inclusive end date in YYYY-MM-DD format.

        Raises:
            KeyError:
                If the ticker does not exist in the price dataset.
        """
        normalized_ticker = self._normalise_ticker(ticker)

        prices = self._load_prices()
        result = prices[prices["ticker"] == normalized_ticker].copy()

        if result.empty:
            raise KeyError(
                f"Ticker '{normalized_ticker}' was not found in the local InvestIQ price dataset."
            )

        if start_date is not None:
            start_ts = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_ts):
                result = result[result["date"] >= start_ts]

        if end_date is not None:
            end_ts = pd.to_datetime(end_date, errors="coerce")
            if pd.notna(end_ts):
                result = result[result["date"] <= end_ts]

        result = result.sort_values("date").reset_index(drop=True)
        return result[self.PRICE_HISTORY_COLUMNS].copy()

    def validate_dataset(self) -> dict[str, Any]:
        """
        Validate loaded datasets and return dataset summary.

        Returns:
            dict:
                Dataset validation status and basic quality statistics.
        """
        fundamentals_missing: list[str] = []
        prices_missing: list[str] = []

        fundamentals_row_count = 0
        price_row_count = 0
        company_count = 0
        year_min = None
        year_max = None
        missing_values_by_column: dict[str, int] = {}

        try:
            fundamentals = self._load_fundamentals()
            fundamentals_missing = [
                col
                for col in self.FUNDAMENTALS_REQUIRED_COLUMNS
                if col not in fundamentals.columns
            ]
        except ValueError:
            fundamentals = pd.DataFrame()
            fundamentals_missing = self.FUNDAMENTALS_REQUIRED_COLUMNS.copy()

        try:
            prices = self._load_prices()
            prices_missing = [
                col
                for col in self.PRICES_REQUIRED_COLUMNS
                if col not in prices.columns
            ]
        except ValueError:
            prices = pd.DataFrame()
            prices_missing = self.PRICES_REQUIRED_COLUMNS.copy()

        if not fundamentals.empty:
            fundamentals_row_count = len(fundamentals)
            company_count = int(fundamentals["ticker"].nunique())
            year_min_value = fundamentals["year"].min()
            year_max_value = fundamentals["year"].max()
            year_min = self._json_safe_value(year_min_value)
            year_max = self._json_safe_value(year_max_value)
            missing_values_by_column = {
                str(col): int(count)
                for col, count in fundamentals.isna().sum().to_dict().items()
            }

        if not prices.empty:
            price_row_count = len(prices)

        missing_required_columns = {
            "fundamentals": fundamentals_missing,
            "prices": prices_missing,
        }

        required_columns_present = (
            len(fundamentals_missing) == 0
            and len(prices_missing) == 0
        )

        return {
            "required_columns_present": required_columns_present,
            "missing_required_columns": missing_required_columns,
            "row_count": fundamentals_row_count,
            "price_row_count": price_row_count,
            "company_count": company_count,
            "year_min": year_min,
            "year_max": year_max,
            "missing_values_by_column": missing_values_by_column,
        }


if __name__ == "__main__":
    loader = StaticDataLoader()
    print(loader.validate_dataset())
