"""
Backtest service for InvestIQ.

Purpose
-------
Run a simple educational backtest using local price data and local annual
fundamentals.

Data sources:
    data/sp500_prices.parquet
    data/sp500_complete.parquet

Optional fallback:
    data/raw/all_stocks_5yr.csv

Rules:
    - No yfinance
    - No SEC EDGAR
    - No Wikipedia
    - No live APIs

Strategy:
    Buy when latest available annual fundamentals at or before the trade year show:
        pe < 15
        roe > 0.15

    Sell:
        after 252 trading days
        or at end_date

Important limitation:
    Fundamentals are annual while prices are daily. This service uses the latest
    available annual fundamentals at or before each trade year. The result is
    educational and illustrative, not investment advice.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from data_layer import StaticDataLoader
except ImportError:
    # Allows imports when used as backend.backtest_service
    from backend.data_layer import StaticDataLoader


class BacktestService:
    """
    Run simple educational valuation-based backtests.

    This service is intentionally simple. It is designed for a Python class
    project and a financial-dashboard frontend, not for production trading.
    """

    HOLDING_PERIOD_DAYS = 252

    PRICE_COLUMNS = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    FUNDAMENTAL_COLUMNS = [
        "ticker",
        "year",
        "period_ending",
        "pe",
        "roe",
        "pb",
        "roa",
        "de",
        "current_ratio",
        "gross_margin",
        "company_name",
        "sector",
        "industry",
    ]

    def __init__(
        self,
        data_loader: StaticDataLoader | None = None,
        fundamentals_path: str = "data/sp500_complete.parquet",
        prices_path: str = "data/sp500_prices.parquet",
        fallback_prices_path: str = "data/raw/all_stocks_5yr.csv",
    ) -> None:
        """
        Create BacktestService.

        Parameters
        ----------
        data_loader:
            Optional StaticDataLoader instance.
        fundamentals_path:
            Path to local complete fundamentals Parquet file.
        prices_path:
            Path to local prices Parquet file.
        fallback_prices_path:
            Optional raw fallback price CSV path.
        """
        self.data_loader = data_loader or StaticDataLoader()

        self.project_root = Path(__file__).resolve().parents[1]
        self.fundamentals_path = self._resolve_path(fundamentals_path)
        self.prices_path = self._resolve_path(prices_path)
        self.fallback_prices_path = self._resolve_path(fallback_prices_path)

        self._fundamentals_df: pd.DataFrame | None = None
        self._prices_df: pd.DataFrame | None = None

    def run_backtest(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 10000,
    ) -> dict[str, Any]:
        """
        Run a simple valuation-rule backtest.

        Parameters
        ----------
        tickers:
            List of ticker symbols.
        start_date:
            Backtest start date in YYYY-MM-DD-compatible format.
        end_date:
            Backtest end date in YYYY-MM-DD-compatible format.
        initial_capital:
            Starting portfolio value.

        Returns
        -------
        dict
            JSON-safe dictionary containing summary, trades, portfolio history,
            metrics, and disclaimer.
        """
        clean_tickers = self._normalize_tickers(tickers)
        start = self._parse_date(start_date, "start_date")
        end = self._parse_date(end_date, "end_date")
        capital = self._validate_initial_capital(initial_capital)

        if start >= end:
            raise ValueError("start_date must be before end_date.")

        fundamentals = self._load_fundamentals()
        prices = self._load_prices()

        prices = prices[
            (prices["ticker"].isin(clean_tickers))
            & (prices["date"] >= start)
            & (prices["date"] <= end)
        ].copy()

        if prices.empty:
            raise ValueError(
                "No price rows found for the requested tickers and date range."
            )

        candidate_trades = self._build_candidate_trades(
            tickers=clean_tickers,
            prices=prices,
            fundamentals=fundamentals,
            start=start,
            end=end,
        )

        executed_trades = self._allocate_and_execute_trades(
            candidate_trades=candidate_trades,
            initial_capital=capital,
        )

        portfolio_history = self._build_portfolio_history(
            prices=prices,
            trades=executed_trades,
            initial_capital=capital,
            start=start,
            end=end,
        )

        metrics = self._calculate_metrics(
            portfolio_history=portfolio_history,
            trades=executed_trades,
            initial_capital=capital,
        )

        summary = self._build_summary(
            tickers=clean_tickers,
            start=start,
            end=end,
            initial_capital=capital,
            trades=executed_trades,
            metrics=metrics,
        )

        return self._json_safe(
            {
                "summary": summary,
                "trades": executed_trades,
                "portfolio_history": portfolio_history,
                "metrics": metrics,
                "methodology": {
                    "strategy": "Buy when latest available annual fundamentals show pe < 15 and roe > 0.15.",
                    "sell_rule": "Sell after 252 trading days or at end_date.",
                    "fundamental_alignment": (
                        "Uses the latest available annual fundamentals at or before "
                        "the trade year. Prices are daily, fundamentals are annual."
                    ),
                    "capital_allocation": (
                        "Initial capital is allocated equally across qualifying trades. "
                        "Cash is held before entry and after exit."
                    ),
                },
                "disclaimer": "Educational backtest. Not investment advice.",
            }
        )

    def _build_candidate_trades(
        self,
        tickers: list[str],
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        """
        Build trade candidates before capital allocation.

        One trade is considered per ticker. The service scans trading days in
        the date range and enters on the first day where the latest available
        annual fundamentals satisfy the buy rule.
        """
        candidates: list[dict[str, Any]] = []

        for ticker in tickers:
            ticker_prices = prices[prices["ticker"] == ticker].copy()

            if ticker_prices.empty:
                candidates.append(
                    self._skipped_trade(
                        ticker=ticker,
                        reason="No price data found in the selected date range.",
                    )
                )
                continue

            ticker_prices = ticker_prices.sort_values("date").reset_index(drop=True)

            ticker_fundamentals = fundamentals[
                fundamentals["ticker"] == ticker
            ].copy()

            if ticker_fundamentals.empty:
                candidates.append(
                    self._skipped_trade(
                        ticker=ticker,
                        reason="No fundamentals found for ticker.",
                    )
                )
                continue

            entry_index = None
            entry_fundamental = None

            for index, price_row in ticker_prices.iterrows():
                trade_year = int(price_row["date"].year)
                latest_fundamental = self._latest_fundamental_at_or_before_year(
                    ticker_fundamentals,
                    trade_year,
                )

                if latest_fundamental is None:
                    continue

                pe = self._normalize_ratio_value("pe", latest_fundamental.get("pe"))
                roe = self._normalize_ratio_value("roe", latest_fundamental.get("roe"))

                if pe is None or roe is None:
                    continue

                if pe < 15 and roe > 0.15:
                    entry_index = index
                    entry_fundamental = latest_fundamental
                    break

            if entry_index is None or entry_fundamental is None:
                candidates.append(
                    self._skipped_trade(
                        ticker=ticker,
                        reason="Buy rule was not triggered in the selected date range.",
                    )
                )
                continue

            exit_index = min(entry_index + self.HOLDING_PERIOD_DAYS, len(ticker_prices) - 1)

            entry_row = ticker_prices.iloc[entry_index]
            exit_row = ticker_prices.iloc[exit_index]

            entry_price = self._to_float_or_none(entry_row.get("close"))
            exit_price = self._to_float_or_none(exit_row.get("close"))

            if entry_price is None or exit_price is None or entry_price <= 0:
                candidates.append(
                    self._skipped_trade(
                        ticker=ticker,
                        reason="Invalid entry or exit close price.",
                    )
                )
                continue

            candidate = {
                "ticker": ticker,
                "company_name": self._json_safe(entry_fundamental.get("company_name")),
                "sector": self._json_safe(entry_fundamental.get("sector")),
                "industry": self._json_safe(entry_fundamental.get("industry")),
                "status": "candidate",
                "entry_date": entry_row["date"],
                "exit_date": exit_row["date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "holding_days": int(exit_index - entry_index),
                "fundamental_year_used": int(entry_fundamental.get("year")),
                "signal_pe": self._normalize_ratio_value("pe", entry_fundamental.get("pe")),
                "signal_roe": self._normalize_ratio_value("roe", entry_fundamental.get("roe")),
                "signal_pb": self._normalize_ratio_value("pb", entry_fundamental.get("pb")),
                "signal_basis": (
                    "latest_available_annual_fundamentals_at_or_before_trade_year"
                ),
                "return_pct": (exit_price / entry_price) - 1.0,
            }

            candidates.append(candidate)

        return candidates

    def _allocate_and_execute_trades(
        self,
        candidate_trades: list[dict[str, Any]],
        initial_capital: float,
    ) -> list[dict[str, Any]]:
        """
        Allocate capital equally across valid candidates and finalize trades.

        Skipped trades remain in the output with status='skipped'.
        """
        executable = [
            trade for trade in candidate_trades
            if trade.get("status") == "candidate"
        ]

        if not executable:
            return [
                {
                    **trade,
                    "allocated_capital": 0.0,
                    "shares": 0.0,
                    "entry_value": 0.0,
                    "exit_value": 0.0,
                    "profit_loss": 0.0,
                }
                for trade in candidate_trades
            ]

        allocation = initial_capital / len(executable)
        executed_trades: list[dict[str, Any]] = []

        for trade in candidate_trades:
            if trade.get("status") != "candidate":
                executed_trades.append(
                    {
                        **trade,
                        "allocated_capital": 0.0,
                        "shares": 0.0,
                        "entry_value": 0.0,
                        "exit_value": 0.0,
                        "profit_loss": 0.0,
                    }
                )
                continue

            entry_price = float(trade["entry_price"])
            exit_price = float(trade["exit_price"])
            shares = allocation / entry_price
            entry_value = shares * entry_price
            exit_value = shares * exit_price
            profit_loss = exit_value - entry_value

            executed_trades.append(
                {
                    **trade,
                    "status": "executed",
                    "allocated_capital": allocation,
                    "shares": shares,
                    "entry_value": entry_value,
                    "exit_value": exit_value,
                    "profit_loss": profit_loss,
                }
            )

        return executed_trades

    def _build_portfolio_history(
        self,
        prices: pd.DataFrame,
        trades: list[dict[str, Any]],
        initial_capital: float,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        """
        Build daily portfolio value history.

        Cash is held before entry and after exit. Positions are valued using
        daily close prices.
        """
        executed_trades = [
            trade for trade in trades
            if trade.get("status") == "executed"
        ]

        if prices.empty:
            return []

        all_dates = (
            prices[
                (prices["date"] >= start)
                & (prices["date"] <= end)
            ]["date"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        if not all_dates:
            return []

        history: list[dict[str, Any]] = []

        for current_date in all_dates:
            portfolio_value = initial_capital

            for trade in executed_trades:
                entry_date = pd.Timestamp(trade["entry_date"])
                exit_date = pd.Timestamp(trade["exit_date"])
                allocated_capital = float(trade["allocated_capital"])
                shares = float(trade["shares"])

                if current_date < entry_date:
                    # Allocation is still cash.
                    continue

                if current_date > exit_date:
                    # Position has been sold and remains as realized cash.
                    portfolio_value += float(trade["profit_loss"])
                    continue

                ticker = trade["ticker"]
                price = self._price_on_or_before(
                    prices=prices,
                    ticker=ticker,
                    current_date=current_date,
                    earliest_date=entry_date,
                )

                if price is None:
                    # If no price is available that day, keep allocated cash.
                    continue

                current_position_value = shares * price
                portfolio_value += current_position_value - allocated_capital

            history.append(
                {
                    "date": current_date,
                    "portfolio_value": portfolio_value,
                    "return_pct": (portfolio_value / initial_capital) - 1.0,
                }
            )

        return history

    @staticmethod
    def _price_on_or_before(
        prices: pd.DataFrame,
        ticker: str,
        current_date: pd.Timestamp,
        earliest_date: pd.Timestamp,
    ) -> float | None:
        """
        Return latest close price on or before current_date, not before entry.
        """
        rows = prices[
            (prices["ticker"] == ticker)
            & (prices["date"] >= earliest_date)
            & (prices["date"] <= current_date)
        ].sort_values("date")

        if rows.empty:
            return None

        value = rows.iloc[-1]["close"]

        try:
            price = float(value)
        except (TypeError, ValueError):
            return None

        if math.isnan(price) or math.isinf(price):
            return None

        return price

    def _calculate_metrics(
        self,
        portfolio_history: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        initial_capital: float,
    ) -> dict[str, Any]:
        """
        Calculate total return, CAGR, max drawdown, and win rate.
        """
        if not portfolio_history:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
            }

        first_date = pd.Timestamp(portfolio_history[0]["date"])
        last_date = pd.Timestamp(portfolio_history[-1]["date"])
        final_value = float(portfolio_history[-1]["portfolio_value"])

        total_return = (final_value / initial_capital) - 1.0

        elapsed_days = max((last_date - first_date).days, 1)
        years = elapsed_days / 365.25
        cagr = (final_value / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0

        values = pd.Series(
            [float(row["portfolio_value"]) for row in portfolio_history],
            dtype="float64",
        )
        running_max = values.cummax()
        drawdowns = (values / running_max) - 1.0
        max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

        executed = [
            trade for trade in trades
            if trade.get("status") == "executed"
        ]

        if executed:
            winning_trades = [
                trade for trade in executed
                if float(trade.get("profit_loss", 0.0)) > 0
            ]
            win_rate = len(winning_trades) / len(executed)
        else:
            win_rate = 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

    def _build_summary(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        initial_capital: float,
        trades: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build summary section for API/frontend display.
        """
        executed_count = sum(1 for trade in trades if trade.get("status") == "executed")
        skipped_count = sum(1 for trade in trades if trade.get("status") == "skipped")

        final_value = initial_capital * (1 + float(metrics.get("total_return", 0.0)))

        return {
            "tickers_requested": tickers,
            "start_date": start,
            "end_date": end,
            "initial_capital": initial_capital,
            "final_value": final_value,
            "executed_trades": executed_count,
            "skipped_trades": skipped_count,
            "strategy_name": "Low P/E and High ROE Annual Fundamentals Strategy",
            "strategy_short_description": (
                "Buy companies with P/E below 15 and ROE above 15%, then hold "
                "for up to 252 trading days."
            ),
        }

    def _latest_fundamental_at_or_before_year(
        self,
        ticker_fundamentals: pd.DataFrame,
        trade_year: int,
    ) -> dict[str, Any] | None:
        """
        Return latest annual fundamental row at or before trade_year.
        """
        eligible = ticker_fundamentals[
            ticker_fundamentals["year"] <= trade_year
        ].copy()

        if eligible.empty:
            return None

        eligible = eligible.sort_values(["year", "period_ending"], na_position="first")
        return eligible.iloc[-1].to_dict()

    def _load_fundamentals(self) -> pd.DataFrame:
        """
        Load complete fundamentals from local Parquet.

        Uses StaticDataLoader-compatible local file paths and validates minimum
        columns needed for this backtest.
        """
        if self._fundamentals_df is not None:
            return self._fundamentals_df.copy()

        if not self.fundamentals_path.exists():
            raise FileNotFoundError(
                f"Missing fundamentals file: {self.fundamentals_path}. "
                "Run: python scripts/build_dataset.py"
            )

        df = pd.read_parquet(self.fundamentals_path)

        for column in self.FUNDAMENTAL_COLUMNS:
            if column not in df.columns:
                df[column] = pd.NA

        df = df[self.FUNDAMENTAL_COLUMNS].copy()
        df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["period_ending"] = pd.to_datetime(df["period_ending"], errors="coerce")

        for metric in ["pe", "roe", "pb", "roa", "de", "current_ratio", "gross_margin"]:
            df[metric] = df[metric].apply(
                lambda value, key=metric: self._normalize_ratio_value(key, value)
            )

        df = df.dropna(subset=["ticker", "year"])
        df["year"] = df["year"].astype(int)
        df = df.sort_values(["ticker", "year", "period_ending"])

        self._fundamentals_df = df.reset_index(drop=True)
        return self._fundamentals_df.copy()

    def _load_prices(self) -> pd.DataFrame:
        """
        Load prices from local Parquet.

        If Parquet is missing and fallback all_stocks_5yr.csv exists, builds a
        temporary price DataFrame from the fallback CSV.
        """
        if self._prices_df is not None:
            return self._prices_df.copy()

        if self.prices_path.exists():
            df = pd.read_parquet(self.prices_path)
        elif self.fallback_prices_path.exists():
            df = self._load_fallback_price_csv(self.fallback_prices_path)
        else:
            raise FileNotFoundError(
                f"Missing price file: {self.prices_path}. "
                "Run: python scripts/build_dataset.py. "
                f"Optional fallback CSV was also not found: {self.fallback_prices_path}"
            )

        for column in self.PRICE_COLUMNS:
            if column not in df.columns:
                raise ValueError(
                    f"Price dataset is missing required column '{column}'. "
                    "Run: python scripts/build_dataset.py"
                )

        df = df[self.PRICE_COLUMNS].copy()
        df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["date", "ticker", "close"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self._prices_df = df
        return self._prices_df.copy()

    def _load_fallback_price_csv(self, path: Path) -> pd.DataFrame:
        """
        Load fallback Kaggle camnugent/sandp500 all_stocks_5yr.csv.

        Expected columns commonly include:
            date, open, high, low, close, volume, Name
        """
        raw = pd.read_csv(path, low_memory=False)
        raw.columns = [str(column).strip() for column in raw.columns]

        rename_map = {}

        if "Name" in raw.columns and "ticker" not in raw.columns:
            rename_map["Name"] = "ticker"

        if "symbol" in raw.columns and "ticker" not in raw.columns:
            rename_map["symbol"] = "ticker"

        if "Symbol" in raw.columns and "ticker" not in raw.columns:
            rename_map["Symbol"] = "ticker"

        raw = raw.rename(columns=rename_map)

        return raw

    @staticmethod
    def _skipped_trade(ticker: str, reason: str) -> dict[str, Any]:
        """
        Build skipped trade record.
        """
        return {
            "ticker": ticker,
            "company_name": None,
            "sector": None,
            "industry": None,
            "status": "skipped",
            "reason": reason,
            "entry_date": None,
            "exit_date": None,
            "entry_price": None,
            "exit_price": None,
            "holding_days": 0,
            "fundamental_year_used": None,
            "signal_pe": None,
            "signal_roe": None,
            "signal_pb": None,
            "signal_basis": None,
            "return_pct": 0.0,
        }

    @staticmethod
    def _normalize_tickers(tickers: list[str]) -> list[str]:
        """
        Normalize ticker list and remove duplicates while preserving order.
        """
        if not tickers:
            raise ValueError("tickers must contain at least one ticker.")

        result: list[str] = []
        seen: set[str] = set()

        for ticker in tickers:
            if ticker is None or not str(ticker).strip():
                continue

            normalized = str(ticker).strip().upper()

            if normalized not in seen:
                result.append(normalized)
                seen.add(normalized)

        if not result:
            raise ValueError("tickers must contain at least one valid ticker.")

        return result

    @staticmethod
    def _parse_date(value: str, field_name: str) -> pd.Timestamp:
        """
        Parse a date string.
        """
        try:
            parsed = pd.Timestamp(value)
        except Exception as exc:
            raise ValueError(f"{field_name} must be a valid date string.") from exc

        if pd.isna(parsed):
            raise ValueError(f"{field_name} must be a valid date string.")

        return parsed.normalize()

    @staticmethod
    def _validate_initial_capital(value: float) -> float:
        """
        Validate initial capital.
        """
        try:
            capital = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("initial_capital must be a positive number.") from exc

        if math.isnan(capital) or math.isinf(capital) or capital <= 0:
            raise ValueError("initial_capital must be a positive number.")

        return capital

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        """
        Convert value to finite float or None.
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

    def _normalize_ratio_value(self, metric: str, value: Any) -> float | None:
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

        if metric in {"roe", "gross_margin"} and abs(number) > 1.5:
            return number / 100.0

        if metric == "current_ratio" and abs(number) > 20:
            return number / 100.0

        return number

    def _resolve_path(self, path_value: str) -> Path:
        """
        Resolve absolute or project-relative paths.
        """
        path = Path(path_value)

        if path.is_absolute():
            return path

        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path

        return self.project_root / path

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
            return value.date().isoformat()

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

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


def run_backtest(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 10000,
) -> dict[str, Any]:
    """
    Convenience function for API modules.
    """
    service = BacktestService()
    return service.run_backtest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )


if __name__ == "__main__":
    service = BacktestService()
    result = service.run_backtest(
        tickers=["AAPL", "MSFT", "IBM", "CSCO"],
        start_date="2014-01-01",
        end_date="2016-12-31",
        initial_capital=10000,
    )

    import json

    print(json.dumps(result, indent=2))
