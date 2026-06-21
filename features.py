# =============================================================
# features.py — the SINGLE place that defines the model's features.
#
# Both model.py (training) and app.py (the web form) read from here,
# so they can never disagree.
#
# 👉 TO ADD A FEATURE:
#    1. If the column already exists in the dataset, just add one entry
#       to the FEATURES list below.
#    2. If the feature must be COMPUTED from a raw column (like floor_level,
#       which we derive from "storey_range"), add one line to clean_data()
#       and then add its entry to FEATURES.
#    That's it — training and the web form update automatically.
# =============================================================

TARGET = "resale_price"


def clean_data(data):
    """Add any derived/computed columns to the raw dataset.

    Add a line here for each feature that isn't already a clean column.
    """
    # Derive a numeric floor from "storey_range" (e.g. "10 TO 12" -> 10)
    data["floor_level"] = data["storey_range"].str.split(" ").str[0].astype(float)

    # --- EXAMPLE (commented out): derive remaining lease in years ---
    # "61 years 04 months" -> 61.0
    # data["remaining_lease_years"] = (
    #     data["remaining_lease"].str.extract(r"(\d+)\s*year").astype(float)
    # )

    return data


# Each feature is one dictionary.
#   type "numeric"     -> shown as a slider; needs min / max / default / step
#   type "categorical" -> shown as a dropdown; choices come from the data
FEATURES = [
    {"col": "floor_area_sqm", "type": "numeric", "label": "Floor area (sqm)",
     "min": 30, "max": 160, "default": 90, "step": 1},
    {"col": "lease_commence_date", "type": "numeric", "label": "Lease commencement year",
     "min": 1970, "max": 2025, "default": 2000, "step": 1},
    {"col": "floor_level", "type": "numeric", "label": "Floor level (storey)",
     "min": 1, "max": 50, "default": 5, "step": 1},
    {"col": "flat_type", "type": "categorical", "label": "Flat type"},
    {"col": "town", "type": "categorical", "label": "Town"},

    # --- EXAMPLE (uncomment after enabling it in clean_data above) ---
    # {"col": "remaining_lease_years", "type": "numeric", "label": "Remaining lease (years)",
    #  "min": 40, "max": 99, "default": 70, "step": 1},
]


def numeric_cols():
    return [f["col"] for f in FEATURES if f["type"] == "numeric"]


def categorical_cols():
    return [f["col"] for f in FEATURES if f["type"] == "categorical"]


def all_cols():
    return numeric_cols() + categorical_cols()
