from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    "pickup",
    "delivery",
    "equipment",
    "distance",
    "weight",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "distance_per_weight",
    "route",
]

CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment", "route"]
NUMERIC_COLUMNS = [
    "distance",
    "weight",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "distance_per_weight",
]


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError(f"Invalid dates in {path}")
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["pickup"] = data["pickup"].astype(str)
    data["delivery"] = data["delivery"].astype(str)
    data["equipment"] = data["equipment"].astype(str)
    data["route"] = data["pickup"] + "->" + data["delivery"]
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["distance_per_weight"] = (
        data["distance"] / (data["weight"].replace(0, np.nan) + 1e-6)
    )
    return data


def build_pipeline() -> Pipeline:
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10,
        sparse_output=False,
    )
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessing = ColumnTransformer(
        [
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
            ("num", numeric_transformer, NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(
        max_iter=600,
        random_state=42,
        learning_rate=0.08,
        max_depth=12,
    )
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", model),
    ])


def train_and_evaluate(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[Pipeline, float, float]:
    pipeline = build_pipeline()
    pipeline.fit(train[FEATURE_COLUMNS], train["posted_rate"])
    valid_pred = pipeline.predict(valid[FEATURE_COLUMNS])
    mae = mean_absolute_error(valid["posted_rate"], valid_pred)
    r2 = r2_score(valid["posted_rate"], valid_pred)
    return pipeline, mae, r2


def split_time_based(frame: pd.DataFrame, train_fraction: float = 0.85) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_data = frame.sort_values("date").reset_index(drop=True)
    split_idx = int(len(sorted_data) * train_fraction)
    train = sorted_data.iloc[:split_idx].reset_index(drop=True)
    valid = sorted_data.iloc[split_idx:].reset_index(drop=True)
    return train, valid


def save_validation_predictions(output_path: Path, validation: pd.DataFrame, predictions: np.ndarray) -> None:
    result = pd.DataFrame(
        {
            "load_id": validation["load_id"].astype(str),
            "predicted_rate": np.maximum(predictions, 1.0),
        }
    )
    result.to_csv(output_path, index=False)


def fill_december_predictions(december_path: Path, output_path: Path, pipeline: Pipeline) -> None:
    december = load_data(december_path)
    december_features = add_features(december)
    predictions = pipeline.predict(december_features[FEATURE_COLUMNS])
    december = december.copy()
    december["predicted_rate"] = np.maximum(predictions, 1.0)
    december["pickup"] = december["pickup"].astype(str)
    december["delivery"] = december["delivery"].astype(str)
    december["equipment"] = december["equipment"].astype(str)
    december = december[
        ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    ]
    december.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the freight rate model and generate predictions.")
    parser.add_argument(
        "--train",
        default="data/train_test.csv",
        help="Labeled training CSV path.",
    )
    parser.add_argument(
        "--validation",
        default="data/validation.csv",
        help="Unlabeled validation CSV path.",
    )
    parser.add_argument(
        "--december",
        default="data/december_chart_inputs.csv",
        help="December chart input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="validation_predictions.csv",
        help="Output CSV for validation predictions.",
    )
    args = parser.parse_args()

    train_df = load_data(Path(args.train))
    validation_df = load_data(Path(args.validation))
    december_df = load_data(Path(args.december))

    train_df = add_features(train_df)
    validation_df = add_features(validation_df)
    december_df = add_features(december_df)

    train_fold, holdout = split_time_based(train_df, train_fraction=0.85)
    model, mae, r2 = train_and_evaluate(train_fold, holdout)
    print(f"Holdout MAE: {mae:.2f}")
    print(f"Holdout R2: {r2:.4f}")

    # Final model uses all training data
    final_model = build_pipeline()
    final_model.fit(train_df[FEATURE_COLUMNS], train_df["posted_rate"])

    validation_preds = final_model.predict(validation_df[FEATURE_COLUMNS])
    save_validation_predictions(Path(args.output), validation_df, validation_preds)
    print(f"Saved validation predictions to {args.output}")

    fill_december_predictions(
        Path(args.december),
        Path(args.december),
        final_model,
    )
    print(f"Filled December predictions in {args.december}")


if __name__ == "__main__":
    main()
