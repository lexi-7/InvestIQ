"""
Tests for backend/data_layer.py.
"""

from __future__ import annotations

import pytest


def test_load_all_companies(data_loader):
    companies = data_loader.load_all_companies()

    assert len(companies) == 5
    assert companies["ticker"].nunique() == 5
    assert set(["ticker", "company_name", "sector", "industry", "year"]).issubset(companies.columns)
    assert companies["year"].min() == 2017
    assert companies["year"].max() == 2017


def test_get_company_by_ticker(data_loader):
    company = data_loader.get_company_by_ticker("aaa")

    assert company["ticker"] == "AAA"
    assert company["company_name"] == "Alpha Analytics Corp."
    assert company["sector"] == "Technology"
    assert company["industry"] == "Software"


def test_unknown_ticker_raises_key_error(data_loader):
    with pytest.raises(KeyError):
        data_loader.get_company_by_ticker("ZZZ")


def test_get_historical_financials(data_loader):
    rows = data_loader.get_historical_financials("AAA", years=2)

    assert len(rows) == 2
    assert list(rows["year"]) == [2016, 2017]
    assert "revenue" in rows.columns
    assert "net_income" in rows.columns


def test_get_ratios(data_loader):
    ratios = data_loader.get_ratios("AAA")

    for key in ["pe", "pb", "roe", "roa", "de", "current_ratio", "gross_margin"]:
        assert key in ratios
        assert ratios[key] is not None

    percentile_keys = [
        key for key in ratios
        if key.endswith("_sector_percentile") or key == "sector_percentiles"
    ]
    assert percentile_keys


def test_get_peer_comparison(data_loader):
    peers = data_loader.get_peer_comparison("AAA", limit=3)

    assert len(peers) >= 2
    assert "AAA" in set(peers["ticker"])
    assert set(["ticker", "company_name", "sector", "industry", "market_cap"]).issubset(peers.columns)


def test_get_price_history(data_loader):
    prices = data_loader.get_price_history(
        "AAA",
        start_date="2015-01-02",
        end_date="2015-01-30",
    )

    assert not prices.empty
    assert set(["date", "open", "high", "low", "close", "volume"]).issubset(prices.columns)
    assert prices["date"].min().strftime("%Y-%m-%d") >= "2015-01-02"
    assert prices["date"].max().strftime("%Y-%m-%d") <= "2015-01-30"
