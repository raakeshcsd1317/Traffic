import argparse
import itertools
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer

warnings.filterwarnings("ignore")

# ── column schema ─────────────────────────────────────────────────────────────

COLUMNS = [
    "Timestamp", "Station", "District", "Freeway", "Direction", "Lane_Type",
    "Station_Length", "Samples", "Percent_Observed", "Total_Flow",
    "Avg_Occupancy", "Avg_Speed",
    "Lane1_Samples", "Lane1_Flow", "Lane1_Avg_Occupancy", "Lane1_Avg_Speed",
    "Lane2_Samples", "Lane2_Flow", "Lane2_Avg_Occupancy", "Lane2_Avg_Speed",
    "Lane3_Samples", "Lane3_Flow", "Lane3_Avg_Occupancy", "Lane3_Avg_Speed",
    "Lane4_Samples", "Lane4_Flow", "Lane4_Avg_Occupancy", "Lane4_Avg_Speed",
    "Lane5_Samples", "Lane5_Flow", "Lane5_Avg_Occupancy", "Lane5_Avg_Speed",
    "Lane6_Samples", "Lane6_Flow", "Lane6_Avg_Occupancy", "Lane6_Avg_Speed",
    "Lane7_Samples", "Lane7_Flow", "Lane7_Avg_Occupancy", "Lane7_Avg_Speed",
    "Lane8_Samples", "Lane8_Flow", "Lane8_Avg_Occupancy", "Lane8_Avg_Speed",
    "Lane9_Samples", "Lane9_Flow", "Lane9_Avg_Occupancy", "Lane9_Avg_Speed",
    "Lane10_Samples", "Lane10_Flow", "Lane10_Avg_Occupancy", "Lane10_Avg_Speed",
]

# Only truly categorical columns get LabelEncoded.
# Hour, Day, Month, Day_of_Week are numeric ordinals — left as integers.
LABEL_COLS = ["Direction", "Lane_Type", "Is_Weekend", "Time_Block"]

TARGET = "Avg_Speed"

# ── helpers ───────────────────────────────────────────────────────────────────

def time_block(hour):
    if hour < 6:   return "0-5"
    if hour < 12:  return "6-11"
    if hour < 18:  return "12-17"
    return "18-23"


def load_and_preprocess(path: str):
    print(f"[1/6] Loading data from {path} …")
    df = pd.read_csv(path, header=None)
    df.columns = COLUMNS

    # ── feature engineering ──────────────────────────────────────────────────
    print("[2/6] Feature engineering …")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Hour"]        = df["Timestamp"].dt.hour.astype("object")
    df["Day"]         = df["Timestamp"].dt.day.astype("object")
    df["Month"]       = df["Timestamp"].dt.month.astype("object")
    df["Day_of_Week"] = df["Timestamp"].dt.dayofweek.astype("object")
    df["Is_Weekend"]  = df["Day_of_Week"].apply(lambda x: "Yes" if int(x) >= 5 else "No")
    df.drop("Timestamp", axis=1, inplace=True)

    # ── drop >90 % null columns ──────────────────────────────────────────────
    print("[3/6] Dropping near-empty columns …")
    null_col = [c for c in df.columns if df[c].isnull().sum() / len(df) * 100 > 90]
    df.drop(null_col, axis=1, inplace=True)

    # ── time-block imputation ────────────────────────────────────────────────
    df["Time_Block"] = df["Hour"].apply(time_block)
    numeric_cols = df.select_dtypes(include="number").columns
    null_num = [c for c in numeric_cols if df[c].isnull().sum() > 0]
    for col in null_num:
        df[col] = df.groupby("Time_Block")[col].transform(lambda x: x.fillna(x.mean()))

    # ── drop highly correlated columns (from notebook analysis) ─────────────
    corr_col = ["Lane1_Avg_Speed", "Lane2_Avg_Speed", "Lane3_Samples",
                "Lane4_Samples", "Lane4_Flow", "District"]
    corr_col = [c for c in corr_col if c in df.columns]
    df.drop(corr_col, axis=1, inplace=True)

    return df


def encode(df, encoders: dict = None, fit: bool = True):
    """Label-encode categorical columns. Returns (df, encoders)."""
    label_cols = [c for c in LABEL_COLS if c in df.columns]
    if fit:
        encoders = {}
        for col in label_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        # Numeric time columns: cast to int (no encoding needed)
        for col in ["Hour", "Day", "Month", "Day_of_Week"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
    else:
        for col in label_cols:
            df[col] = encoders[col].transform(df[col].astype(str))
        for col in ["Hour", "Day", "Month", "Day_of_Week"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
    return df, encoders


def transform_skew(df, transformer=None, skew_cols=None, fit: bool = True):
    """Yeo-Johnson transform skewed numeric features."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)

    if fit:
        skew_cols = [c for c in numeric_cols if df[c].skew() > 1.5]
        transformer = PowerTransformer()
        if skew_cols:
            df[skew_cols] = transformer.fit_transform(df[skew_cols])
    else:
        if skew_cols:
            df[skew_cols] = transformer.transform(df[skew_cols])
    return df, transformer, skew_cols


# ── main ──────────────────────────────────────────────────────────────────────

def main(data_path: str, output_dir: str = "artifacts"):
    Path(output_dir).mkdir(exist_ok=True)

    df = load_and_preprocess(data_path)

    print("[4/6] Encoding & transforming …")
    df, encoders = encode(df, fit=True)
    df, pt, skew_cols = transform_skew(df, fit=True)

    # ── train / test split ───────────────────────────────────────────────────
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # save feature order for inference
    feature_names = X_train.columns.tolist()

    # ── train best model (Random Forest) ────────────────────────────────────
    print("[5/6] Training RandomForestRegressor (this may take a few minutes) …")
    model = RandomForestRegressor(n_estimators=50,max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_pred = model.predict(X_test)
    print("\n── Evaluation on held-out test set ──")
    print(f"  R²   : {r2_score(y_test, y_pred):.4f}")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f}")

    # ── save artifacts ───────────────────────────────────────────────────────
    print(f"\n[6/6] Saving artifacts to '{output_dir}/' …")
    with open(f"{output_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{output_dir}/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open(f"{output_dir}/power_transformer.pkl", "wb") as f:
        pickle.dump(pt, f)
    with open(f"{output_dir}/skew_cols.pkl", "wb") as f:
        pickle.dump(skew_cols, f)
    with open(f"{output_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("Done ✓  All artifacts saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to the raw .txt data file")
    parser.add_argument("--output", default="artifacts",
                        help="Folder to save model artifacts (default: artifacts/)")
    args = parser.parse_args()
    main(args.data, args.output)