"""
FastAPI application for InvestIQ.

Purpose
-------
Expose offline-first financial analysis endpoints using local Parquet datasets.

Data source:
    data/sp500_complete.parquet
    data/sp500_prices.parquet

These Parquet files are created from local CSV source files:
    data/raw/fundamentals.csv
    data/raw/securities.csv
    data/raw/prices-split-adjusted.csv

Rules:
    - No yfinance
    - No SEC EDGAR
    - No Wikipedia
    - No Kaggle API at runtime
    - No live network calls
    - Do not train ML at startup
    - Load ML model lazily only when /api/ml_classify is called
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

try:
    from data_layer import StaticDataLoader
    from ratio_service import RatioService
    from peer_service import PeerComparisonService
    from ml_service import MLValuationService
    from backtest_service import BacktestService
except ImportError:
    # Allows running as: uvicorn backend.app:app
    from backend.data_layer import StaticDataLoader
    from backend.ratio_service import RatioService
    from backend.peer_service import PeerComparisonService
    from backend.ml_service import MLValuationService
    from backend.backtest_service import BacktestService


APP_VERSION = "2.0.0"
TICKER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    """
    Request body for POST /api/backtest.
    """

    tickers: list[str] = Field(
        ...,
        min_length=1,
        description="Ticker symbols to include in the backtest.",
        examples=[["AAPL", "MSFT", "IBM"]],
    )
    start_date: str = Field(
        ...,
        description="Backtest start date, for example 2014-01-01.",
        examples=["2014-01-01"],
    )
    end_date: str = Field(
        ...,
        description="Backtest end date, for example 2016-12-31.",
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
        """
        Validate ticker list.
        """
        if not value:
            raise ValueError("At least one ticker is required.")

        normalized: list[str] = []
        seen: set[str] = set()

        for ticker in value:
            clean = normalize_ticker(ticker)
            if clean not in seen:
                normalized.append(clean)
                seen.add(clean)

        return normalized


# -----------------------------------------------------------------------------
# App and services
# -----------------------------------------------------------------------------

app = FastAPI(
    title="InvestIQ API",
    version=APP_VERSION,
    description=(
        "Offline-first financial analysis API using local source-file-derived "
        "Parquet datasets."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_loader = StaticDataLoader()
ratio_service = RatioService(data_loader=data_loader)
peer_service = PeerComparisonService(data_loader=data_loader)
backtest_service = BacktestService(data_loader=data_loader)

# ML service is intentionally lazy. It is created only when /api/ml_classify is used.
_ml_service: MLValuationService | None = None


# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add request processing time header.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.6f}"
    return response


# -----------------------------------------------------------------------------
# Response helpers
# -----------------------------------------------------------------------------

def utc_timestamp() -> str:
    """
    Return current UTC timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def success_response(data: Any) -> dict[str, Any]:
    """
    Build standard success envelope.
    """
    return {
        "success": True,
        "data": json_safe(data),
        "timestamp": utc_timestamp(),
        "version": APP_VERSION,
    }


def error_response(error: str, code: int) -> JSONResponse:
    """
    Build standard error envelope.
    """
    return JSONResponse(
        status_code=code,
        content={
            "success": False,
            "error": error,
            "code": code,
            "timestamp": utc_timestamp(),
            "version": APP_VERSION,
        },
    )


def json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy values and DataFrames into JSON-safe Python values.
    """
    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]

    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def normalize_ticker(ticker: str) -> str:
    """
    Validate and normalize ticker symbols.
    """
    if ticker is None or not str(ticker).strip():
        raise ValueError("Ticker is required.")

    clean = str(ticker).strip().upper()

    if not TICKER_PATTERN.match(clean):
        raise ValueError(
            "Invalid ticker format. Use letters, numbers, dot, or hyphen only."
        )

    return clean


def get_ml_service() -> MLValuationService:
    """
    Lazily create MLValuationService.

    This prevents model loading or model checks during API startup.
    """
    global _ml_service

    if _ml_service is None:
        _ml_service = MLValuationService(data_loader=data_loader)

    return _ml_service


def dataset_missing_message() -> str:
    """
    Standard dataset-missing message.
    """
    return "Dataset not found. Run python scripts/build_dataset.py first."


# -----------------------------------------------------------------------------
# Exception handlers
# -----------------------------------------------------------------------------

@app.exception_handler(FileNotFoundError)
async def handle_file_not_found(_: Request, __: FileNotFoundError) -> JSONResponse:
    """
    Return clear dataset missing error.
    """
    return error_response(dataset_missing_message(), 500)


@app.exception_handler(KeyError)
async def handle_key_error(_: Request, exc: KeyError) -> JSONResponse:
    """
    Return 404 for unknown ticker.
    """
    message = str(exc).strip("'\"") or "Ticker not found."
    return error_response(message, 404)


@app.exception_handler(ValueError)
async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
    """
    Return 400 for bad input.
    """
    return error_response(str(exc), 400)


@app.exception_handler(HTTPException)
async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
    """
    Return standard envelope for FastAPI HTTPException.
    """
    return error_response(str(exc.detail), exc.status_code)


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    """
    Return standard envelope for unexpected errors.
    """
    return error_response(f"Unexpected server error: {exc}", 500)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, Any]:
    """
    API and dataset health check.
    """
    try:
        dataset_validation = data_loader.validate_dataset()
        dataset_status = "ready"
    except FileNotFoundError:
        dataset_validation = {
            "required_columns_present": False,
            "message": dataset_missing_message(),
        }
        dataset_status = "missing"

    return success_response(
        {
            "status": "ok",
            "dataset_status": dataset_status,
            "offline_mode": True,
            "data_source": {
                "fundamentals": "data/sp500_complete.parquet",
                "prices": "data/sp500_prices.parquet",
                "created_from": [
                    "data/raw/fundamentals.csv",
                    "data/raw/securities.csv",
                    "data/raw/prices-split-adjusted.csv",
                ],
            },
            "ml_loading": "lazy",
            "dataset_validation": dataset_validation,
        }
    )


@app.get("/api/autocomplete")
def autocomplete(
    q: str = Query("", description="Ticker or company-name search text."),
    limit: int = Query(12, ge=1, le=50),
) -> dict[str, Any]:
    """
    Return ticker/company autocomplete matches.
    """
    query = str(q or "").strip().upper()
    companies = data_loader.load_all_companies().copy()

    if companies.empty:
        return success_response([])

    for column in ["ticker", "company_name", "sector", "industry"]:
        if column not in companies.columns:
            companies[column] = pd.NA

    companies["ticker"] = companies["ticker"].astype("string").str.strip().str.upper()
    companies["company_name"] = companies["company_name"].astype("string").str.strip()

    if query:
        ticker_match = companies["ticker"].str.contains(query, case=False, na=False)
        name_match = companies["company_name"].str.contains(query, case=False, na=False)
        companies = companies[ticker_match | name_match]

    companies = companies.sort_values(["ticker"]).head(limit)

    results = [
        {
            "ticker": row.get("ticker"),
            "company_name": row.get("company_name"),
            "sector": row.get("sector"),
            "industry": row.get("industry"),
            "label": f"{row.get('ticker')} - {row.get('company_name')}",
        }
        for row in companies.to_dict(orient="records")
    ]

    return success_response(results)


@app.get("/api/company_profile")
def company_profile(
    ticker: str = Query(..., description="Ticker symbol, for example AAPL."),
) -> dict[str, Any]:
    """
    Return latest company profile for a ticker.
    """
    clean_ticker = normalize_ticker(ticker)
    company = data_loader.get_company_by_ticker(clean_ticker)
    return success_response(company)


@app.get("/api/historical_financials")
def historical_financials(
    ticker: str = Query(..., description="Ticker symbol, for example AAPL."),
    years: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """
    Return annual historical financial rows for a ticker.
    """
    clean_ticker = normalize_ticker(ticker)
    financials = data_loader.get_historical_financials(clean_ticker, years=years)
    return success_response(financials)


@app.get("/api/financial_ratios")
def financial_ratios(
    ticker: str = Query(..., description="Ticker symbol, for example AAPL."),
) -> dict[str, Any]:
    """
    Return beginner-friendly financial ratio analysis for a ticker.
    """
    clean_ticker = normalize_ticker(ticker)
    result = ratio_service.analyze_ticker(clean_ticker)
    return success_response(result)


@app.get("/api/peer_comparison")
def peer_comparison(
    ticker: str = Query(..., description="Ticker symbol, for example AAPL."),
    limit: int = Query(8, ge=1, le=50),
) -> dict[str, Any]:
    """
    Return target company, peers, averages, and chart-ready peer data.
    """
    clean_ticker = normalize_ticker(ticker)
    result = peer_service.get_peer_comparison_payload(clean_ticker, limit=limit)
    return success_response(result)


@app.get("/api/ml_classify")
def ml_classify(
    ticker: str = Query(..., description="Ticker symbol, for example AAPL."),
) -> dict[str, Any]:
    """
    Return ML valuation classification.

    The trained model is loaded lazily here, not at API startup.
    If model files are missing, MLValuationService uses rule-based fallback.
    """
    clean_ticker = normalize_ticker(ticker)
    service = get_ml_service()
    result = service.classify_ticker(clean_ticker)
    return success_response(result)


@app.post("/api/backtest")
def backtest(payload: BacktestRequest) -> dict[str, Any]:
    """
    Run educational backtest.
    """
    result = backtest_service.run_backtest(
        tickers=payload.tickers,
        start_date=payload.start_date,
        end_date=payload.end_date,
        initial_capital=payload.initial_capital,
    )

    return success_response(result)


# -----------------------------------------------------------------------------
# Local development entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
