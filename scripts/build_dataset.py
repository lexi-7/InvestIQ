"""
Build local Parquet datasets for InvestIQ.

Purpose
-------
Converts Kaggle CSV files into clean, offline-first Parquet files:

Input:
    data/raw/fundamentals.csv
    data/raw/securities.csv
    data/raw/prices-split-adjusted.csv

Optional fallback:
    data/raw/all_stocks_5yr.csv

Output:
    data/sp500_complete.parquet
    data/sp500_prices.parquet

Rules:
    - No yfinance
    - No SEC EDGAR
    - No Wikipedia
    - No live API calls
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

FUNDAMENTALS_PATH = RAW_DIR / "fundamentals.csv"
SECURITIES_PATH = RAW_DIR / "securities.csv"
PRIMARY_PRICES_PATH = RAW_DIR / "prices-split-adjusted.csv"
FALLBACK_PRICES_PATH = RAW_DIR / "all_stocks_5yr.csv"

COMPLETE_OUTPUT_PATH = DATA_DIR / "sp500_complete.parquet"
PRICES_OUTPUT_PATH = DATA_DIR / "sp500_prices.parquet"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger("investiq.build_dataset")


# -----------------------------------------------------------------------------
# Column mapping configuration
# -----------------------------------------------------------------------------

FUNDAMENTAL_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "ticker": [
        "Ticker Symbol",
        "Ticker symbol",
        "ticker symbol",
        "Symbol",
        "symbol",
        "Ticker",
        "ticker",
    ],
    "year": [
        "For Year",
        "for year",
        "Year",
        "year",
        "Fiscal Year",
        "fiscal year",
    ],
    "period_ending": [
        "Period Ending",
        "period ending",
        "Period End",
        "period end",
        "Date",
        "date",
    ],
    "revenue": [
        "Total Revenue",
        "total revenue",
        "Revenue",
        "revenue",
    ],
    "gross_profit": [
        "Gross Profit",
        "gross profit",
    ],
    "net_income": [
        "Net Income",
        "net income",
        "Net Income Applicable To Common Shares",
    ],
    "total_assets": [
        "Total Assets",
        "total assets",
    ],
    "total_equity": [
        "Total Equity",
        "total equity",
        "Total Stockholder Equity",
        "Total Stockholders Equity",
        "Stockholders Equity",
        "Shareholders Equity",
    ],
    "total_debt": [
        "Long-Term Debt",
        "Long Term Debt",
        "long-term debt",
        "long term debt",
        "Total Debt",
        "total debt",
    ],
    "current_assets": [
        "Total Current Assets",
        "total current assets",
        "Current Assets",
        "current assets",
    ],
    "current_liabilities": [
        "Total Current Liabilities",
        "total current liabilities",
        "Current Liabilities",
        "current liabilities",
    ],
    "eps": [
        "Earnings Per Share",
        "earnings per share",
        "EPS",
        "eps",
    ],
    "estimated_shares": [
        "Estimated Shares Outstanding",
        "estimated shares outstanding",
        "Shares Outstanding",
        "shares outstanding",
    ],
    "current_ratio": [
        "Current Ratio",
        "current ratio",
    ],
    "gross_margin": [
        "Gross Margin",
        "gross margin",
    ],
    "roe": [
        "After Tax ROE",
        "after tax roe",
        "ROE",
        "roe",
        "Return on Equity",
        "return on equity",
    ],
}

SECURITIES_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "ticker": [
        "Ticker symbol",
        "Ticker Symbol",
        "ticker symbol",
        "Symbol",
        "symbol",
        "Ticker",
        "ticker",
    ],
    "company_name": [
        "Security",
        "security",
        "Company",
        "company",
        "Company Name",
        "company name",
        "Name",
        "name",
    ],
    "sector": [
        "GICS Sector",
        "gics sector",
        "Sector",
        "sector",
    ],
    "industry": [
        "GICS Sub Industry",
        "GICS Sub-Industry",
        "gics sub industry",
        "GICS Industry",
        "industry",
        "Industry",
        "Sub Industry",
        "sub industry",
    ],
}

PRICE_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "date": ["date", "Date"],
    "ticker": ["symbol", "Symbol", "ticker", "Ticker", "Name", "name"],
    "open": ["open", "Open"],
    "high": ["high", "High"],
    "low": ["low", "Low"],
    "close": ["close", "Close"],
    "volume": ["volume", "Volume"],
}

FUNDAMENTAL_OUTPUT_COLUMNS = [
    "ticker",
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
    "current_ratio",
    "gross_margin",
    "roe",
]

SECURITIES_OUTPUT_COLUMNS = [
    "ticker",
    "company_name",
    "sector",
    "industry",
]

PRICE_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

NUMERIC_COLUMNS = [
    "year",
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
    "current_ratio",
    "gross_margin",
    "roe",
    "roa",
    "de",
    "year_end_price",
    "market_cap",
    "pe",
    "pb",
]

NON_IMPUTED_COLUMNS = [
    "ticker",
    "company_name",
    "sector",
    "industry",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_column_name(name: str) -> str:
    """Normalize a column name for resilient matching."""
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def available_columns_text(columns: Iterable[str]) -> str:
    return "\n".join(f"  - {col}" for col in columns)


def stop_with_column_error(
    file_path: Path,
    required_standard_columns: list[str],
    mapped_columns: dict[str, str],
    available_columns: Iterable[str],
) -> None:
    missing = [col for col in required_standard_columns if col not in mapped_columns]
    logger.error("Cannot map required columns in %s", file_path)
    logger.error("Missing standard columns: %s", ", ".join(missing))
    logger.error("Available columns:\n%s", available_columns_text(available_columns))
    raise SystemExit(1)


def build_column_map(
    available_columns: Iterable[str],
    candidates: dict[str, list[str]],
) -> dict[str, str]:
    """
    Return mapping from source column name to standard column name.

    Example return:
        {"Ticker Symbol": "ticker", "Total Revenue": "revenue"}
    """
    available = list(available_columns)
    normalized_available = {
        normalize_column_name(col): col
        for col in available
    }

    source_to_standard: dict[str, str] = {}

    for standard_name, possible_names in candidates.items():
        for possible_name in possible_names:
            normalized_possible = normalize_column_name(possible_name)
            if normalized_possible in normalized_available:
                source_col = normalized_available[normalized_possible]
                source_to_standard[source_col] = standard_name
                break

    return source_to_standard


def require_file(path: Path, explanation: str) -> None:
    if not path.exists():
        logger.error("Missing required input file: %s", path)
        logger.error("Place %s at: %s", explanation, path)
        raise SystemExit(1)


def read_csv(path: Path) -> pd.DataFrame:
    logger.info("Reading %s", path)
    try:
        return strip_column_names(pd.read_csv(path, low_memory=False))
    except Exception as exc:
        logger.error("Failed to read %s", path)
        logger.error("Reason: %s", exc)
        raise SystemExit(1) from exc


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    logger.info("Writing %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except ImportError as exc:
        logger.error("pyarrow is required to write Parquet files.")
        logger.error("Install it with: pip install pyarrow")
        raise SystemExit(1) from exc
    except Exception as exc:
        logger.error("Failed to write %s", path)
        logger.error("Reason: %s", exc)
        raise SystemExit(1) from exc


def clean_ticker(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
    )


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: pd.NA})
    return numerator / denominator


def add_missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def first_non_null_by_row(left: pd.Series, right: pd.Series) -> pd.Series:
    """Use left when present; otherwise use right."""
    return left.where(left.notna(), right)


# -----------------------------------------------------------------------------
# Loading and transformation
# -----------------------------------------------------------------------------

def load_fundamentals() -> pd.DataFrame:
    require_file(
        FUNDAMENTALS_PATH,
        "Kaggle dgawlik/nyse fundamentals.csv",
    )

    raw = read_csv(FUNDAMENTALS_PATH)
    column_map = build_column_map(raw.columns, FUNDAMENTAL_COLUMN_CANDIDATES)

    required = ["ticker"]
    if not any(target in column_map.values() for target in ["year", "period_ending"]):
        logger.error("Cannot map either year or period_ending in %s", FUNDAMENTALS_PATH)
        logger.error("At least one of these must exist: For Year, Year, Period Ending, Date")
        logger.error("Available columns:\n%s", available_columns_text(raw.columns))
        raise SystemExit(1)

    reverse_map = {standard: source for source, standard in column_map.items()}
    mapped_standard_names = set(reverse_map.keys())
    if any(col not in mapped_standard_names for col in required):
        stop_with_column_error(FUNDAMENTALS_PATH, required, reverse_map, raw.columns)

    df = raw.rename(columns=column_map)
    df = add_missing_columns(df, FUNDAMENTAL_OUTPUT_COLUMNS)
    df = df[FUNDAMENTAL_OUTPUT_COLUMNS].copy()

    df["ticker"] = clean_ticker(df["ticker"])
    df["period_ending"] = pd.to_datetime(df["period_ending"], errors="coerce")

    df = to_numeric(df, NUMERIC_COLUMNS)

    derived_year = df["period_ending"].dt.year

    if "year" in df.columns:
        df["year"] = first_non_null_by_row(df["year"], derived_year)
    else:
        df["year"] = derived_year

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Fix malformed fiscal years from raw Kaggle data.
    # Example: a bad value like 1215 should be replaced by the year from Period Ending.
    current_year = pd.Timestamp.today().year
    invalid_year_mask = (
        df["year"].isna()
        | (df["year"] < 1990)
        | (df["year"] > current_year)
    )
    df.loc[invalid_year_mask, "year"] = derived_year.loc[invalid_year_mask]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    logger.info("Loaded fundamentals rows: %s", len(df))
    return df


def load_securities() -> pd.DataFrame:
    require_file(
        SECURITIES_PATH,
        "Kaggle dgawlik/nyse securities.csv",
    )

    raw = read_csv(SECURITIES_PATH)
    column_map = build_column_map(raw.columns, SECURITIES_COLUMN_CANDIDATES)

    reverse_map = {standard: source for source, standard in column_map.items()}
    required = ["ticker"]
    if any(col not in reverse_map for col in required):
        stop_with_column_error(SECURITIES_PATH, required, reverse_map, raw.columns)

    df = raw.rename(columns=column_map)
    df = add_missing_columns(df, SECURITIES_OUTPUT_COLUMNS)
    df = df[SECURITIES_OUTPUT_COLUMNS].copy()

    df["ticker"] = clean_ticker(df["ticker"])

    for col in ["company_name", "sector", "industry"]:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    df = df.dropna(subset=["ticker"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")

    logger.info("Loaded securities rows: %s", len(df))
    return df


def resolve_price_input_path() -> Path:
    if PRIMARY_PRICES_PATH.exists():
        return PRIMARY_PRICES_PATH

    if FALLBACK_PRICES_PATH.exists():
        logger.warning(
            "Primary price file not found: %s. Using fallback: %s",
            PRIMARY_PRICES_PATH,
            FALLBACK_PRICES_PATH,
        )
        return FALLBACK_PRICES_PATH

    logger.error("Missing price input file.")
    logger.error("Place Kaggle dgawlik/nyse prices-split-adjusted.csv at: %s", PRIMARY_PRICES_PATH)
    logger.error("Optional fallback: place Kaggle camnugent/sandp500 all_stocks_5yr.csv at: %s", FALLBACK_PRICES_PATH)
    raise SystemExit(1)


def load_prices() -> pd.DataFrame:
    price_path = resolve_price_input_path()
    raw = read_csv(price_path)
    column_map = build_column_map(raw.columns, PRICE_COLUMN_CANDIDATES)

    reverse_map = {standard: source for source, standard in column_map.items()}
    required = ["date", "ticker", "close"]
    if any(col not in reverse_map for col in required):
        stop_with_column_error(price_path, required, reverse_map, raw.columns)

    df = raw.rename(columns=column_map)
    df = add_missing_columns(df, PRICE_OUTPUT_COLUMNS)
    df = df[PRICE_OUTPUT_COLUMNS].copy()

    df["ticker"] = clean_ticker(df["ticker"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    price_numeric_columns = ["open", "high", "low", "close", "volume"]
    df = to_numeric(df, price_numeric_columns)

    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info("Loaded price rows: %s", len(df))
    return df


def calculate_year_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    annual_prices = prices.copy()
    annual_prices["year"] = annual_prices["date"].dt.year

    annual_prices = annual_prices.dropna(subset=["ticker", "year", "date", "close"])
    annual_prices = annual_prices.sort_values(["ticker", "year", "date"])

    year_end = (
        annual_prices.groupby(["ticker", "year"], as_index=False)
        .tail(1)[["ticker", "year", "close"]]
        .rename(columns={"close": "year_end_price"})
    )

    year_end["year"] = pd.to_numeric(year_end["year"], errors="coerce")
    return year_end.reset_index(drop=True)


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    calculated_roe = safe_divide(df["net_income"], df["total_equity"])
    calculated_roa = safe_divide(df["net_income"], df["total_assets"])
    calculated_de = safe_divide(df["total_debt"], df["total_equity"])
    calculated_current_ratio = safe_divide(df["current_assets"], df["current_liabilities"])
    calculated_gross_margin = safe_divide(df["gross_profit"], df["revenue"])

    df["roe"] = first_non_null_by_row(df["roe"], calculated_roe)
    df["roa"] = calculated_roa
    df["de"] = calculated_de
    df["current_ratio"] = first_non_null_by_row(df["current_ratio"], calculated_current_ratio)
    df["gross_margin"] = first_non_null_by_row(df["gross_margin"], calculated_gross_margin)

    df["market_cap"] = df["year_end_price"] * df["estimated_shares"]
    df["pe"] = safe_divide(df["year_end_price"], df["eps"])
    df["pb"] = safe_divide(df["market_cap"], df["total_equity"])

    return df


def clean_complete_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ticker"] = clean_ticker(df["ticker"])
    df["period_ending"] = pd.to_datetime(df["period_ending"], errors="coerce")
    df = to_numeric(df, NUMERIC_COLUMNS)

    df = df.dropna(subset=["ticker", "year"])
    df["year"] = df["year"].astype(int)

    # Remove duplicate ticker/year rows.
    # Prefer the row with the latest period_ending when available.
    df = df.sort_values(["ticker", "year", "period_ending"], na_position="first")
    df = df.drop_duplicates(subset=["ticker", "year"], keep="last")

    df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
    return df


def impute_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute numeric values by sector median first, then global median.

    Text identity/profile fields are not imputed:
        ticker, company_name, sector, industry
    """
    df = df.copy()

    numeric_columns_to_impute = [
        col for col in df.select_dtypes(include=["number"]).columns
        if col not in NON_IMPUTED_COLUMNS
    ]

    for col in numeric_columns_to_impute:
        sector_median = df.groupby("sector", dropna=False)[col].transform("median")
        global_median = df[col].median(skipna=True)
        df[col] = df[col].fillna(sector_median)
        df[col] = df[col].fillna(global_median)

    return df


def print_data_quality_report(
    complete_before_imputation: pd.DataFrame,
    complete_after_imputation: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    missing_before = complete_before_imputation.isna().sum().sort_values(ascending=False)
    missing_after = complete_after_imputation.isna().sum().sort_values(ascending=False)

    company_count = complete_after_imputation["ticker"].nunique()
    row_count = len(complete_after_imputation)
    min_year = complete_after_imputation["year"].min()
    max_year = complete_after_imputation["year"].max()
    companies_with_sector = complete_after_imputation.loc[
        complete_after_imputation["sector"].notna(),
        "ticker",
    ].nunique()
    companies_with_industry = complete_after_imputation.loc[
        complete_after_imputation["industry"].notna(),
        "ticker",
    ].nunique()

    print("\n" + "=" * 80)
    print("InvestIQ Data Quality Report")
    print("=" * 80)
    print(f"Number of companies: {company_count}")
    print(f"Number of rows: {row_count}")
    print(f"Year range: {min_year} - {max_year}")
    print(f"Companies with sector: {companies_with_sector}")
    print(f"Companies with industry: {companies_with_industry}")
    print(f"Number of price rows: {len(prices)}")

    print("\nMissing values by column before imputation:")
    print(missing_before.to_string())

    print("\nMissing values by column after imputation:")
    print(missing_after.to_string())
    print("=" * 80 + "\n")


def build_dataset() -> None:
    logger.info("Starting InvestIQ dataset build")
    logger.info("Project root: %s", PROJECT_ROOT)

    fundamentals = load_fundamentals()
    securities = load_securities()
    prices = load_prices()

    save_parquet(prices, PRICES_OUTPUT_PATH)

    year_end_prices = calculate_year_end_prices(prices)

    complete = fundamentals.merge(
        securities,
        on="ticker",
        how="left",
    )

    complete = complete.merge(
        year_end_prices,
        on=["ticker", "year"],
        how="left",
    )

    complete = clean_complete_dataset(complete)
    complete = add_ratios(complete)
    complete = clean_complete_dataset(complete)

    missing_before_imputation = complete.copy()
    complete = impute_numeric_values(complete)

    # Put identity/profile columns first, then analytical columns.
    preferred_column_order = [
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
        "current_ratio",
        "gross_margin",
        "roe",
        "roa",
        "de",
        "pe",
        "pb",
    ]

    complete = add_missing_columns(complete, preferred_column_order)
    complete = complete[preferred_column_order]

    save_parquet(complete, COMPLETE_OUTPUT_PATH)

    print_data_quality_report(
        complete_before_imputation=missing_before_imputation,
        complete_after_imputation=complete,
        prices=prices,
    )

    logger.info("Dataset build completed successfully")
    logger.info("Created: %s", COMPLETE_OUTPUT_PATH)
    logger.info("Created: %s", PRICES_OUTPUT_PATH)


def main() -> None:
    try:
        build_dataset()
    except KeyboardInterrupt:
        logger.error("Dataset build interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
