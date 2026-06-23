# =============================================================
# features.py — defines the model's features AND the selectable model
# "profiles" (Old, Enhanced B, Enhanced C, Enhanced D).
#
# Both model.py (training) and app.py (the web form + selector) read from
# here, so they can never disagree. Each profile mirrors a step from the
# C32_practice notebook.
# =============================================================

TARGET = "resale_price"


def clean_data(data):
    """Add every derived/computed column the profiles might need.

    Mirrors the cleaning done in the C32_practice notebook.
    """
    # Derive a numeric floor from "storey_range" (e.g. "10 TO 12" -> 10)
    data["floor_level"] = data["storey_range"].str.split(" ").str[0].astype(float)

    # remaining_lease "61 years 04 months" -> 61.33 (years)
    def _lease_to_years(s):
        s = str(s)
        years = 0.0
        if "year" in s:
            years += float(s.split("year")[0].strip())
        if "month" in s:
            digits = "".join(ch for ch in s.split("year")[-1] if ch.isdigit())
            if digits:
                years += float(digits) / 12
        return years

    if "remaining_lease" in data.columns:
        data["remaining_lease_years"] = data["remaining_lease"].apply(_lease_to_years)

    # Transaction year from the "month" column ("2017-03" -> 2017)
    if "month" in data.columns:
        data["txn_year"] = data["month"].str.split("-").str[0].astype(int)

    return data


# ---- The catalogue of every feature any profile can use ----
#   type "numeric"     -> slider (needs min / max / default / step)
#   type "categorical" -> dropdown (choices come from the data)
FEATURE_DEFS = {
    "floor_area_sqm": {"type": "numeric", "label": "Floor area (sqm)",
                       "min": 30, "max": 160, "default": 90, "step": 1},
    "lease_commence_date": {"type": "numeric", "label": "Lease commencement year",
                            "min": 1970, "max": 2025, "default": 2000, "step": 1},
    "floor_level": {"type": "numeric", "label": "Floor level (storey)",
                    "min": 1, "max": 50, "default": 5, "step": 1},
    "remaining_lease_years": {"type": "numeric", "label": "Remaining lease (years)",
                              "min": 40, "max": 99, "default": 70, "step": 1},
    "txn_year": {"type": "numeric", "label": "Transaction year",
                 "min": 2017, "max": 2025, "default": 2024, "step": 1},
    "flat_type": {"type": "categorical", "label": "Flat type"},
    "town": {"type": "categorical", "label": "Town"},
    "flat_model": {"type": "categorical", "label": "Flat model"},
}

# Shared feature groups (mirror the notebook)
_PARTB_FEATURES = ["floor_area_sqm", "lease_commence_date", "floor_level",
                   "flat_type", "town"]
_ENHANCED_FEATURES = ["floor_area_sqm", "lease_commence_date", "floor_level",
                      "remaining_lease_years", "txn_year",
                      "flat_type", "town", "flat_model"]


# ---- The selectable model profiles (each = one notebook step) ----
PROFILES = {
    "old": {
        "label": "🅾️ Old model — baseline (3 features)",
        "short": "Old model",
        "cols": ["floor_area_sqm", "lease_commence_date", "floor_level"],
        "estimator": "linear",
        "desc": "The original L05 model: floor area, lease year, floor level. "
                "Linear Regression. A weak-but-honest starting point.",
    },
    "enhanced_b": {
        "label": "🅱️ Enhanced B — more features",
        "short": "Enhanced B",
        "cols": _PARTB_FEATURES,
        "estimator": "linear",
        "desc": "Adds flat type & town (one-hot encoded). Same Linear Regression — "
                "more relevant information, lower error.",
    },
    "enhanced_c": {
        "label": "🅲 Enhanced C — tuned Gradient Boosting",
        "short": "Enhanced C",
        "cols": _ENHANCED_FEATURES,
        "estimator": "hist_gb_tuned",
        "desc": "8 engineered features (incl. remaining-lease & flat model) with a "
                "tuned HistGradientBoosting model. Strong AND barely overfits.",
    },
    "enhanced_d": {
        "label": "🅳 Enhanced D — stacking ensemble",
        "short": "Enhanced D",
        "cols": _ENHANCED_FEATURES,
        "estimator": "stack",
        "desc": "A diverse team — Random Forest + tuned HistGB + Ridge, blended by a "
                "RidgeCV manager. The best scores in the lab (at ~3x the cost).",
    },
}

DEFAULT_PROFILE = "enhanced_c"


def profile_features(key):
    """Return the list of feature dicts (with 'col' merged in) for a profile."""
    feats = []
    for col in PROFILES[key]["cols"]:
        feats.append({"col": col, **FEATURE_DEFS[col]})
    return feats


def categorical_cols(key):
    return [c for c in PROFILES[key]["cols"]
            if FEATURE_DEFS[c]["type"] == "categorical"]


def all_categorical_cols():
    """Every categorical column used by any profile (for building dropdowns)."""
    cols = []
    for key in PROFILES:
        for c in categorical_cols(key):
            if c not in cols:
                cols.append(c)
    return cols
