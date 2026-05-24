"""
Pytest fixtures for InvestIQ.

No internet. No yfinance. No Kaggle API.
Tests use small generated temporary Parquet files.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def sample_parquet_files(tmp_path: Path) -> dict[str, Path]:
    """
    Build a small local test dataset:
    - 5 companies
    - 2 sectors
    - 2 industries
    - 3 fiscal years
    - simple price history
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    fundamentals_path = data_dir / "sp500_complete.parquet"
    prices_path = data_dir / "sp500_prices.parquet"

    companies = [
        ("AAA", "Alpha Analytics Corp.", "Technology", "Software", 50_000_000_000, 10_000_000_000, 2_000_000_000, 20_000_000_000, 9_000_000_000, 6_000_000_000, 2_500_000_000, [13, 14, 12], [2.0, 2.2, 2.4], [0.20, 0.21, 0.24], [0.09, 0.10, 0.11], [0.35, 0.38, 0.40], [1.8, 1.9, 2.0], [0.55, 0.56, 0.58]),
        ("BBB", "Beta Business Systems", "Technology", "Software", 35_000_000_000, 8_000_000_000, 1_100_000_000, 16_000_000_000, 7_000_000_000, 4_200_000_000, 2_800_000_000, [18, 19, 20], [2.8, 3.0, 3.1], [0.13, 0.14, 0.15], [0.06, 0.07, 0.07], [0.55, 0.60, 0.65], [1.3, 1.4, 1.5], [0.40, 0.41, 0.42]),
        ("CCC", "Core Cloud Co.", "Technology", "Software", 25_000_000_000, 6_000_000_000, 600_000_000, 12_000_000_000, 5_000_000_000, 2_100_000_000, 4_500_000_000, [28, 30, 32], [4.2, 4.4, 4.6], [0.06, 0.07, 0.07], [0.02, 0.025, 0.03], [1.2, 1.3, 1.4], [0.9, 1.0, 1.1], [0.28, 0.30, 0.31]),
        ("DDD", "Delta Devices Inc.", "Industrials", "Machinery", 15_000_000_000, 12_000_000_000, 900_000_000, 18_000_000_000, 8_000_000_000, 4_000_000_000, 3_000_000_000, [14, 15, 16], [1.4, 1.5, 1.6], [0.11, 0.12, 0.13], [0.04, 0.045, 0.05], [0.45, 0.48, 0.50], [1.6, 1.7, 1.8], [0.32, 0.33, 0.34]),
        ("EEE", "Epsilon Equipment Ltd.", "Industrials", "Machinery", 10_000_000_000, 9_000_000_000, 300_000_000, 14_000_000_000, 4_000_000_000, 2_200_000_000, 6_000_000_000, [24, 26, 28], [3.2, 3.4, 3.6], [0.06, 0.07, 0.07], [0.015, 0.02, 0.02], [1.6, 1.7, 1.8], [0.8, 0.9, 0.95], [0.18, 0.19, 0.20]),
    ]

    years = [2015, 2016, 2017]
    rows: list[dict[str, Any]] = []

    for company in companies:
        (
            ticker, company_name, sector, industry, market_cap_base, revenue_base,
            net_income_base, total_assets_base, total_equity_base, gross_profit_base,
            total_debt_base, pe_values, pb_values, roe_values, roa_values, de_values,
            current_ratio_values, gross_margin_values
        ) = company

        for index, year in enumerate(years):
            growth = 1 + index * 0.08
            estimated_shares = 1_000_000_000
            year_end_price = market_cap_base * growth / estimated_shares

            rows.append({
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "industry": industry,
                "year": year,
                "period_ending": pd.Timestamp(f"{year}-12-31"),
                "revenue": revenue_base * growth,
                "gross_profit": gross_profit_base * growth,
                "net_income": net_income_base * growth,
                "total_assets": total_assets_base * growth,
                "total_equity": total_equity_base * growth,
                "total_debt": total_debt_base * growth,
                "current_assets": 3_000_000_000 * growth,
                "current_liabilities": 1_800_000_000 * growth,
                "eps": 2.0 + index,
                "estimated_shares": estimated_shares,
                "year_end_price": year_end_price,
                "market_cap": market_cap_base * growth,
                "pe": pe_values[index],
                "pb": pb_values[index],
                "roe": roe_values[index],
                "roa": roa_values[index],
                "de": de_values[index],
                "current_ratio": current_ratio_values[index],
                "gross_margin": gross_margin_values[index],
            })

    pd.DataFrame(rows).to_parquet(fundamentals_path, index=False)

    price_rows: list[dict[str, Any]] = []
    dates = pd.bdate_range("2015-01-02", "2018-01-31", freq="B")

    for company_index, company in enumerate(companies):
        ticker = company[0]
        base_price = 40 + company_index * 10
        for day_index, date in enumerate(dates):
            close = base_price + day_index * 0.02 + company_index
            price_rows.append({
                "date": date,
                "ticker": ticker,
                "open": close * 0.995,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000 + company_index * 10_000,
            })

    pd.DataFrame(price_rows).to_parquet(prices_path, index=False)

    return {
        "fundamentals_path": fundamentals_path,
        "prices_path": prices_path,
        "data_dir": data_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture()
def data_loader(sample_parquet_files):
    from backend.data_layer import StaticDataLoader

    return StaticDataLoader(
        fundamentals_path=str(sample_parquet_files["fundamentals_path"]),
        prices_path=str(sample_parquet_files["prices_path"]),
    )


@pytest.fixture()
def test_client(sample_parquet_files, monkeypatch):
    """
    FastAPI TestClient wired to the generated temporary Parquet files.
    """
    import backend.app as app_module
    from backend.backtest_service import BacktestService
    from backend.data_layer import StaticDataLoader
    from backend.peer_service import PeerComparisonService
    from backend.ratio_service import RatioService

    loader = StaticDataLoader(
        fundamentals_path=str(sample_parquet_files["fundamentals_path"]),
        prices_path=str(sample_parquet_files["prices_path"]),
    )

    monkeypatch.setattr(app_module, "data_loader", loader)
    monkeypatch.setattr(app_module, "ratio_service", RatioService(data_loader=loader))
    monkeypatch.setattr(app_module, "peer_service", PeerComparisonService(data_loader=loader))
    monkeypatch.setattr(app_module, "backtest_service", BacktestService(
        data_loader=loader,
        fundamentals_path=str(sample_parquet_files["fundamentals_path"]),
        prices_path=str(sample_parquet_files["prices_path"]),
        fallback_prices_path=str(sample_parquet_files["tmp_path"] / "missing.csv"),
    ))
    monkeypatch.setattr(app_module, "_ml_service", None)

    return TestClient(app_module.app)
