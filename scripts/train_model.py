"""
Train ML valuation classifier for InvestIQ.

Purpose
-------
Train a local RandomForest valuation classifier using only:

    data/sp500_complete.parquet

Outputs:
    models/valuation_model.joblib
    models/scaler.joblib
    models/model_metrics.json

Rules:
    - No yfinance
    - No SEC EDGAR
    - No Wikipedia
    - No live APIs
    - No Kaggle API at runtime
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sp500_complete.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "valuation_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger("investiq.train_model")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

FEATURES = [
    "pe",
    "pb",
    "roe",
    "roa",
    "de",
    "current_ratio",
    "gross_margin",
]

REQUIRED_COLUMNS = [
    "ticker",
    "sector",
    *FEATURES,
]

LABEL_UNDERVALUED = "UNDERVALUED"
LABEL_FAIRLY_VALUED = "FAIRLY_VALUED"
LABEL_OVERVALUED = "OVERVALUED"

MIN_TRAINING_ROWS = 50
TEST_SIZE = 0.25
RANDOM_STATE = 42


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def require_input_file(path: Path) -> None:
    """
    Validate that the training dataset exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training dataset: {path}\n"
            "Run this first from the project root:\n"
            "    python scripts/build_dataset.py"
        )


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate required columns.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        available = "\n".join(f"  - {column}" for column in df.columns)
        missing_text = ", ".join(missing)
        raise ValueError(
            "The training dataset is missing required columns.\n"
            f"Missing: {missing_text}\n"
            "Available columns:\n"
            f"{available}\n"
            "Rebuild the dataset with:\n"
            "    python scripts/build_dataset.py"
        )


def to_float_or_none(value: Any) -> float | None:
    """
    Convert value to a finite float or None.
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


def normalize_ratio_value(metric: str, value: Any) -> float | None:
    """
    Normalize known Kaggle ratio scale inconsistencies.

    Examples found in dgawlik/nyse:
        roe = 36 means 0.36
        gross_margin = 39 means 0.39
        current_ratio = 135 means 1.35
    """
    number = to_float_or_none(value)
    if number is None:
        return None

    if metric in {"roe", "gross_margin"} and abs(number) > 1.5:
        return number / 100.0

    if metric == "current_ratio" and abs(number) > 20:
        return number / 100.0

    return number


def clean_metric_for_training(metric: str, value: Any) -> float | None:
    """
    Remove invalid or extreme metric values before training and labeling.

    The goal is not to make the data perfect. The goal is to prevent obvious
    broken values, such as negative P/E, from dominating sector medians and
    model behavior.
    """
    number = normalize_ratio_value(metric, value)

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


def load_training_data() -> pd.DataFrame:
    """
    Load and prepare local Parquet training data.
    """
    require_input_file(DATA_PATH)

    logger.info("Reading %s", DATA_PATH)

    try:
        df = pd.read_parquet(DATA_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {DATA_PATH}. "
            "Rebuild the dataset with: python scripts/build_dataset.py"
        ) from exc

    validate_columns(df)

    df = df.copy()

    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    df["sector"] = df["sector"].astype("string").str.strip()

    for feature in FEATURES:
        df[feature] = df[feature].apply(
            lambda value, metric=feature: clean_metric_for_training(metric, value)
        )

    df = df.dropna(subset=["ticker", "sector"])

    # Impute remaining feature gaps by sector median, then global median.
    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

        sector_median = df.groupby("sector", dropna=False)[feature].transform("median")
        global_median = df[feature].median(skipna=True)

        df[feature] = df[feature].fillna(sector_median)
        df[feature] = df[feature].fillna(global_median)

    df = df.dropna(subset=FEATURES)

    if len(df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Too few training rows after cleaning: {len(df)}. "
            f"Minimum required: {MIN_TRAINING_ROWS}. "
            "Check data/sp500_complete.parquet or rebuild the dataset."
        )

    return df.reset_index(drop=True)


def add_sector_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sector-relative statistics used for label creation.
    """
    df = df.copy()

    sector_group = df.groupby("sector", dropna=False)

    df["sector_median_pe"] = sector_group["pe"].transform("median")
    df["sector_p75_pe"] = sector_group["pe"].transform(lambda series: series.quantile(0.75))
    df["sector_median_pb"] = sector_group["pb"].transform("median")
    df["sector_p75_pb"] = sector_group["pb"].transform(lambda series: series.quantile(0.75))
    df["sector_median_roe"] = sector_group["roe"].transform("median")
    df["sector_median_roa"] = sector_group["roa"].transform("median")

    return df


def create_label(row: pd.Series) -> str:
    """
    Create valuation label using sector-relative valuation rules.

    UNDERVALUED:
        pe < sector median pe
        pb <= sector median pb
        roe > sector median roe

    OVERVALUED:
        pe > sector 75th percentile pe
        OR pb > sector 75th percentile pb AND roa < sector median roa
        OR roe < sector median roe AND pe > sector median pe

    FAIRLY_VALUED:
        all other cases
    """
    is_undervalued = (
        row["pe"] < row["sector_median_pe"]
        and row["pb"] <= row["sector_median_pb"]
        and row["roe"] > row["sector_median_roe"]
    )

    if is_undervalued:
        return LABEL_UNDERVALUED

    is_overvalued = (
        row["pe"] > row["sector_p75_pe"]
        or (
            row["pb"] > row["sector_p75_pb"]
            and row["roa"] < row["sector_median_roa"]
        )
        or (
            row["roe"] < row["sector_median_roe"]
            and row["pe"] > row["sector_median_pe"]
        )
    )

    if is_overvalued:
        return LABEL_OVERVALUED

    return LABEL_FAIRLY_VALUED


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target valuation labels.
    """
    labeled = add_sector_statistics(df)
    labeled["valuation_label"] = labeled.apply(create_label, axis=1)

    label_counts = labeled["valuation_label"].value_counts().to_dict()

    if len(label_counts) < 2:
        raise ValueError(
            "Only one valuation class was created. "
            f"Label distribution: {label_counts}. "
            "The model needs at least two classes."
        )

    return labeled


def can_use_stratify(labels: pd.Series) -> bool:
    """
    Use stratified split only when every class has at least 2 rows.
    """
    label_counts = labels.value_counts()

    if len(label_counts) < 2:
        return False

    return bool((label_counts >= 2).all())


def train_model(df: pd.DataFrame) -> dict[str, Any]:
    """
    Train RandomForestClassifier and return model metrics.
    """
    X = df[FEATURES]
    y = df["valuation_label"]

    stratify = y if can_use_stratify(y) else None

    if stratify is None:
        logger.warning(
            "Not using stratified split because one or more classes has too few rows."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    labels = sorted(y.unique().tolist())

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    feature_importances = {
        feature: float(importance)
        for feature, importance in zip(FEATURES, model.feature_importances_)
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
        "feature_importances": feature_importances,
        "training_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_distribution": {
            str(label): int(count)
            for label, count in y.value_counts().to_dict().items()
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "model_type": "RandomForestClassifier",
        "scaler_type": "StandardScaler",
        "target": "valuation_label",
        "label_rules": {
            LABEL_UNDERVALUED: [
                "pe < sector median pe",
                "pb <= sector median pb",
                "roe > sector median roe",
            ],
            LABEL_OVERVALUED: [
                "pe > sector 75th percentile pe",
                "pb > sector 75th percentile pb AND roa < sector median roa",
                "roe < sector median roe AND pe > sector median pe",
            ],
            LABEL_FAIRLY_VALUED: [
                "all other cases",
            ],
        },
        "data_source": str(DATA_PATH.relative_to(PROJECT_ROOT)),
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(metrics), file, indent=2)

    return metrics


def make_json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


def print_training_summary(metrics: dict[str, Any]) -> None:
    """
    Print concise training results to console.
    """
    print("\n" + "=" * 80)
    print("InvestIQ ML Training Summary")
    print("=" * 80)

    print(f"Training row count: {metrics['training_rows']}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")

    print("\nLabel distribution:")
    for label, count in metrics["label_distribution"].items():
        print(f"  {label}: {count}")

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")

    print("\nFeature importances:")
    sorted_importances = sorted(
        metrics["feature_importances"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for feature, importance in sorted_importances:
        print(f"  {feature}: {importance:.4f}")

    print("\nSaved files:")
    print(f"  {MODEL_PATH}")
    print(f"  {SCALER_PATH}")
    print(f"  {METRICS_PATH}")
    print("=" * 80 + "\n")


def main() -> None:
    """
    Main entry point.
    """
    try:
        df = load_training_data()
        df = create_labels(df)
        metrics = train_model(df)
        print_training_summary(metrics)

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    except KeyboardInterrupt:
        logger.error("Training interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        logger.exception("Unexpected training failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
