"""
Tests for FastAPI endpoints.
"""

from __future__ import annotations


def assert_success(response):
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "data" in payload
    assert "timestamp" in payload
    assert payload["version"] == "2.0.0"
    return payload["data"]


def test_api_health(test_client):
    response = test_client.get("/api/health")
    data = assert_success(response)

    assert data["status"] == "ok"
    assert data["dataset_status"] == "ready"
    assert data["offline_mode"] is True


def test_api_company_profile(test_client):
    response = test_client.get("/api/company_profile?ticker=AAA")
    data = assert_success(response)

    assert data["ticker"] == "AAA"
    assert data["company_name"] == "Alpha Analytics Corp."


def test_api_historical_financials(test_client):
    response = test_client.get("/api/historical_financials?ticker=AAA&years=2")
    data = assert_success(response)

    assert len(data) == 2
    assert data[-1]["year"] == 2017


def test_api_financial_ratios(test_client):
    response = test_client.get("/api/financial_ratios?ticker=AAA")
    data = assert_success(response)

    assert data["ticker"] == "AAA"
    assert "ratios" in data
    assert len(data["ratios"]) == 7
    assert "financial_health_score" in data


def test_api_peer_comparison(test_client):
    response = test_client.get("/api/peer_comparison?ticker=AAA&limit=3")
    data = assert_success(response)

    assert data["target"]["ticker"] == "AAA"
    assert len(data["peers"]) <= 3
    assert "bar_chart_data" in data
    assert "radar_chart_data" in data


def test_invalid_ticker(test_client):
    response = test_client.get("/api/company_profile?ticker=BAD!!!")

    assert response.status_code == 400
    payload = response.json()
    assert payload["success"] is False
    assert payload["code"] == 400
    assert "Invalid ticker format" in payload["error"]


def test_unknown_ticker_returns_404(test_client):
    response = test_client.get("/api/company_profile?ticker=ZZZ")

    assert response.status_code == 404
    payload = response.json()
    assert payload["success"] is False
    assert payload["code"] == 404
    assert "ZZZ" in payload["error"]
