# =============================================================
# L06 — Better, Trustworthy HDB Price Model
# Trains ONE model per selectable profile (Old / Enhanced B / C / D),
# defined in features.py, and bundles them together for the app.
#
# Run standalone to (re)train and save the model file:
#       python model.py
# =============================================================

import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

import features as ft

DATA_URL = (
    "https://raw.githubusercontent.com/kohjiaxuan/"
    "Predicting-HDB-Price-with-Machine-Learning/master/"
    "resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_model.pkl")


def load_data():
    """Download the live HDB resale dataset and apply the cleaning steps."""
    data = pd.read_csv(DATA_URL)
    data = ft.clean_data(data)
    return data


def make_estimator(kind):
    """Map a profile's estimator key to a fresh scikit-learn model."""
    if kind == "linear":
        return LinearRegression()
    if kind == "hist_gb_tuned":
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.1, max_depth=8,
            l2_regularization=1.0, random_state=42,
        )
    if kind == "stack":
        return StackingRegressor(
            estimators=[
                ("random_forest", RandomForestRegressor(
                    n_estimators=150, random_state=42, n_jobs=-1)),
                ("hist_gb_tuned", HistGradientBoostingRegressor(
                    max_iter=500, learning_rate=0.1, max_depth=8,
                    l2_regularization=1.0, random_state=42)),
                ("ridge", RidgeCV()),
            ],
            final_estimator=RidgeCV(),
            passthrough=True, cv=3, n_jobs=-1,
        )
    raise ValueError(f"Unknown estimator kind: {kind}")


def train_profile(data, key):
    """Train one profile's model and return its bundle dict."""
    profile = ft.PROFILES[key]
    cat_cols = ft.categorical_cols(key)

    X = pd.get_dummies(data[profile["cols"]], columns=cat_cols)
    y = data[ft.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_estimator(profile["estimator"])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model,
        "columns": list(X.columns),
        "label": profile["label"],
        "short": profile["short"],
        "desc": profile["desc"],
        "mae": mean_absolute_error(y_test, preds),
        "mape": mean_absolute_percentage_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }


def train_all():
    """Train every profile and return one combined bundle."""
    data = load_data()

    profiles = {key: train_profile(data, key) for key in ft.PROFILES}

    # Dropdown choices for every categorical column (shared across profiles)
    categories = {
        col: sorted(data[col].dropna().unique().tolist())
        for col in ft.all_categorical_cols()
    }

    return {
        "profiles": profiles,
        "categories": categories,
        "n_rows": len(data),
        "default_profile": ft.DEFAULT_PROFILE,
    }


def save_model(bundle, path=MODEL_PATH):
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_or_train(path=MODEL_PATH):
    """Load the saved bundle, or train and save one if none exists yet."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        # Re-train if an old single-model pickle is found (schema changed)
        if "profiles" in bundle:
            return bundle
    bundle = train_all()
    save_model(bundle, path)
    return bundle


if __name__ == "__main__":
    print("Downloading data and training all model profiles...")
    bundle = train_all()
    print(f"\nTrained on {bundle['n_rows']:,} rows.\n")
    header = f"{'Profile':14s} {'MAE':>12s} {'MAPE':>7s} {'R2':>7s}"
    print(header)
    print("-" * len(header))
    for key, p in bundle["profiles"].items():
        print(f"{p['short']:14s} S${p['mae']:>10,.0f} {p['mape']:>6.1%} {p['r2']:>7.3f}")
    save_model(bundle)
    print(f"\nSaved model bundle to {MODEL_PATH}")
