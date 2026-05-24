"""
Reusable Pydantic schemas for InvestIQ FastAPI backend.

Purpose
-------
Centralize request and response schemas used by backend/app.py and service
modules.

The schemas are intentionally practical:
    - strict enough for validation
    - flexible enough for the current service payloads
    - compatible with the existing offline-first InvestIQ backend

Rules:
    - No live APIs
    - No yfinance
    - No SEC EDGAR
    - No Wikipedia
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


APP_VERSION = "2.0.0"
TICKER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


# -----------------------------------------------------------------------------
# Shared validation helpers
# -----------------------------------------------------------------------------

def normalize_ticker_value(value: str) -> str:
    """
    Normalize and validate ticker symbol.

    Valid examples:
        AAPL
        BRK.B
        BRK-B
        MSFT
    """
    if value is None or not str(value).strip():
        raise ValueError("Ticker is required.")

    ticker = str(value).strip().upper()

    if not TICKER_PATTERN.match(ticker):
        raise ValueError(
            "Invalid ticker format. Use letters, numbers, dot, or hyphen only."
        )

    return ticker


# -----------------------------------------------------------------------------
# Standard API envelopes
# -----------------------------------------------------------------------------

class SuccessEnvelope(BaseModel):
    """
    Standard success response envelope.

    Example:
        {
          "success": true,
          "data": {...},
          "timestamp": "...",
          "version": "2.0.0"
        }
    """

    success: Literal[True] = True
    data: Any
    timestamp: str
    version: str = APP_VERSION


class ErrorEnvelope(BaseModel):
    """
    Standard error response envelope.

    Example:
        {
          "success": false,
          "error": "...",
          "code": 400,
          "timestamp": "...",
          "version": "2.0.0"
        }
    """

    success: Literal[False] = False
    error: str
    code: int
    timestamp: str
    version: str = APP_VERSION


# -----------------------------------------------------------------------------
# Common query/request models
# -----------------------------------------------------------------------------

class TickerQuery(BaseModel):
    """
    Shared ticker query model.
    """

    ticker: str = Field(..., examples=["AAPL"])

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalize_ticker_value(value)


class AutocompleteQuery(BaseModel):
    """
    Autocomplete query model.
    """

    q: str = Field("", description="Ticker or company-name search text.")
    limit: int = Field(12, ge=1, le=50)


class HistoricalFinancialsQuery(TickerQuery):
    """
    Historical financials query model.
    """

    years: int = Field(5, ge=1, le=20)


class PeerComparisonQuery(TickerQuery):
    """
    Peer comparison query model.
    """

    limit: int = Field(8, ge=1, le=50)


class BacktestRequest(BaseModel):
    """
    Request body for POST /api/backtest.
    """

    tickers: list[str] = Field(
        ...,
        min_length=1,
        description="Ticker symbols to include in the backtest.",
        examples=[["AAPL", "MSFT", "IBM", "CSCO"]],
    )
    start_date: str = Field(
        ...,
        description="Backtest start date.",
        examples=["2014-01-01"],
    )
    end_date: str = Field(
        ...,
        description="Backtest end date.",
        examples=["2016-12-31"],
    )
    initial_capital: float = Field(
        10000,
        gt=0,
        description="Initial capital for the educational backtest.",
        examples=[10000],
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one ticker is required.")

        normalized: list[str] = []
        seen: set[str] = set()

        for ticker in value:
            clean = normalize_ticker_value(ticker)

            if clean not in seen:
                normalized.append(clean)
                seen.add(clean)

        return normalized


# -----------------------------------------------------------------------------
# Company and financial data schemas
# -----------------------------------------------------------------------------

class CompanyProfile(BaseModel):
    """
    Latest company profile and core financial metrics.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    year: int | None = None

    revenue: float | None = None
    net_income: float | None = None
    total_assets: float | None = None
    total_equity: float | None = None
    year_end_price: float | None = None
    market_cap: float | None = None

    pe: float | None = None
    pb: float | None = None
    roe: float | None = None
    roa: float | None = None
    de: float | None = None
    current_ratio: float | None = None
    gross_margin: float | None = None


class AutocompleteItem(BaseModel):
    """
    One autocomplete result.
    """

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    label: str


class HistoricalFinancialRow(BaseModel):
    """
    One annual financial history row.
    """

    model_config = ConfigDict(extra="allow")

    year: int
    revenue: float | None = None
    net_income: float | None = None
    total_assets: float | None = None
    total_equity: float | None = None
    gross_profit: float | None = None
    total_debt: float | None = None


# -----------------------------------------------------------------------------
# Ratio schemas
# -----------------------------------------------------------------------------

RatioStatus = Literal["green", "yellow", "red", "grey"]


class RatioCard(BaseModel):
    """
    Beginner-friendly ratio card.
    """

    key: str
    label: str
    value: float | None = None
    formatted_value: str
    status: RatioStatus
    explanation: str
    good_range: str
    bad_range: str
    sector_percentile: float | None = None


class RatioAnalysisPayload(BaseModel):
    """
    Response data from RatioService.
    """

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    ratios: list[RatioCard]
    financial_health_score: int = Field(..., ge=0, le=100)
    summary: str


# -----------------------------------------------------------------------------
# Peer comparison schemas
# -----------------------------------------------------------------------------

class PeerCompany(BaseModel):
    """
    Company row used in peer comparison.
    """

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None

    pe: float | None = None
    pb: float | None = None
    roe: float | None = None
    roa: float | None = None
    de: float | None = None
    current_ratio: float | None = None
    gross_margin: float | None = None


class PeerChartData(BaseModel):
    """
    Bar-chart or radar-chart data.

    The service may include extra fields such as:
        note
        scale
        higher_is_better
        lower_is_better
        average_method
    """

    model_config = ConfigDict(extra="allow")

    metric_keys: list[str]
    labels: list[str]
    target: list[float | None]
    peer_average: list[float | None]
    sector_average: list[float | None]
    industry_average: list[float | None]


class PeerComparisonPayload(BaseModel):
    """
    Full peer comparison response data.
    """

    target: PeerCompany
    peers: list[PeerCompany]
    sector_averages: dict[str, float | None]
    industry_averages: dict[str, float | None]
    bar_chart_data: PeerChartData
    radar_chart_data: PeerChartData


# -----------------------------------------------------------------------------
# ML valuation schemas
# -----------------------------------------------------------------------------

ValuationVerdict = Literal["UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"]


class FeatureValue(BaseModel):
    """
    Formatted feature value.
    """

    label: str
    value: float | None = None
    formatted_value: str


class FeatureImportanceItem(BaseModel):
    """
    Feature importance item from the model metrics.
    """

    feature: str
    label: str
    importance: float
    value: float | None = None
    formatted_value: str


class ValuationSignal(BaseModel):
    """
    Positive or negative valuation signal.
    """

    feature: str
    label: str
    value: float | None = None
    formatted_value: str
    message: str


class MLClassificationPayload(BaseModel):
    """
    ML classification response data.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    company_name: str | None = None
    verdict: ValuationVerdict
    confidence: float | None = Field(None, ge=0, le=1)
    probabilities: dict[ValuationVerdict, float]
    model_accuracy: float | None = Field(None, ge=0, le=1)
    classifier_type: Literal["random_forest", "rule_based_fallback"]
    feature_importance: list[FeatureImportanceItem]
    feature_values: dict[str, FeatureValue]
    top_positive_signals: list[ValuationSignal]
    top_negative_signals: list[ValuationSignal]
    explanation: str


# -----------------------------------------------------------------------------
# Backtest schemas
# -----------------------------------------------------------------------------

class BacktestSummary(BaseModel):
    """
    Backtest summary section.
    """

    tickers_requested: list[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    executed_trades: int
    skipped_trades: int
    strategy_name: str
    strategy_short_description: str


class BacktestTrade(BaseModel):
    """
    One backtest trade or skipped ticker.
    """

    model_config = ConfigDict(extra="allow")

    ticker: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    status: Literal["executed", "skipped", "candidate"] | str
    reason: str | None = None

    entry_date: str | None = None
    exit_date: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    holding_days: int | None = None

    fundamental_year_used: int | None = None
    signal_pe: float | None = None
    signal_roe: float | None = None
    signal_pb: float | None = None
    signal_basis: str | None = None

    return_pct: float | None = None
    allocated_capital: float | None = None
    shares: float | None = None
    entry_value: float | None = None
    exit_value: float | None = None
    profit_loss: float | None = None


class PortfolioHistoryPoint(BaseModel):
    """
    One portfolio-history chart point.
    """

    date: str
    portfolio_value: float
    return_pct: float


class BacktestMetrics(BaseModel):
    """
    Backtest metrics section.
    """

    total_return: float
    cagr: float
    max_drawdown: float
    win_rate: float


class BacktestMethodology(BaseModel):
    """
    Backtest methodology text used by the frontend.
    """

    strategy: str
    sell_rule: str
    fundamental_alignment: str
    capital_allocation: str


class BacktestPayload(BaseModel):
    """
    Full backtest response data.
    """

    summary: BacktestSummary
    trades: list[BacktestTrade]
    portfolio_history: list[PortfolioHistoryPoint]
    metrics: BacktestMetrics
    methodology: BacktestMethodology
    disclaimer: str


# -----------------------------------------------------------------------------
# Health schemas
# -----------------------------------------------------------------------------

class DataSourceInfo(BaseModel):
    """
    Dataset source metadata.
    """

    fundamentals: str
    prices: str
    created_from: list[str]


class DatasetValidation(BaseModel):
    """
    Dataset validation result from StaticDataLoader.
    """

    model_config = ConfigDict(extra="allow")

    required_columns_present: bool
    missing_required_columns: dict[str, list[str]] | None = None
    row_count: int | None = None
    price_row_count: int | None = None
    company_count: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    missing_values_by_column: dict[str, int] | None = None
    message: str | None = None


class HealthPayload(BaseModel):
    """
    Health endpoint response data.
    """

    status: str
    dataset_status: Literal["ready", "missing"] | str
    offline_mode: bool
    data_source: DataSourceInfo
    ml_loading: str
    dataset_validation: DatasetValidation
