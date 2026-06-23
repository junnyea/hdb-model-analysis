# =============================================================
# L06 — HDB Resale Price Predictor (deployable Streamlit app)
#
# Pick a MODEL PROFILE in the sidebar — Old, Enhanced B, C or D — each
# mirroring a step from the C32_practice notebook. The input form rebuilds
# itself to match the chosen profile's features (defined in features.py).
#
# Run locally:
#       pip install -r requirements.txt
#       streamlit run app.py
# =============================================================

import pandas as pd
import streamlit as st

import features as ft
from model import load_or_train

st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏡", layout="centered")


@st.cache_resource(show_spinner="Training the models (first run only, ~1-2 min)…")
def get_models():
    """Load the saved bundle, or train all profiles on first run. Cached."""
    return load_or_train()


bundle = get_models()
profiles = bundle["profiles"]
categories = bundle["categories"]

# ---- Header ----
st.title("🏡 Singapore HDB Resale Price Predictor")
st.caption(
    f"Trained on {bundle['n_rows']:,} real resale transactions (2017 onwards). "
    "Choose a model in the sidebar to compare the notebook's steps live."
)

# ---- Model profile selector ----
keys = list(profiles.keys())
default_key = bundle.get("default_profile", keys[0])
if default_key not in keys:  # the default profile may have failed to train
    default_key = keys[0]
choice = st.sidebar.radio(
    "Model",
    keys,
    index=keys.index(default_key),
    format_func=lambda k: profiles[k]["label"],
)
prof = profiles[choice]
st.sidebar.caption(prof["desc"])

# ---- Chosen model report card ----
st.subheader(prof["label"])
st.write(prof["desc"])
m1, m2, m3 = st.columns(3)
m1.metric("Average error (MAE)", f"S${prof['mae']:,.0f}")
m2.metric("MAPE", f"{prof['mape']:.1%}")
m3.metric("R²", f"{prof['r2']:.3f}")

with st.expander("📊 Compare all four models", expanded=False):
    st.table(
        pd.DataFrame(
            [{"Model": p["short"],
              "MAE": f"S${p['mae']:,.0f}",
              "MAPE": f"{p['mape']:.1%}",
              "R²": f"{p['r2']:.3f}"}
             for p in profiles.values()]
        )
    )
    st.caption(
        "Tested on flats the models never saw. More features (B), a tuned model "
        "(C), and a diverse ensemble (D) each lower the error."
    )

# ---- Inputs (rebuilt from the chosen profile's features) ----
st.sidebar.header("Flat details")
user_input = {}
for f in ft.profile_features(choice):
    col, label = f["col"], f["label"]
    if f["type"] == "numeric":
        user_input[col] = st.sidebar.slider(
            label, f["min"], f["max"], f["default"], f["step"], key=f"in_{col}"
        )
    else:  # categorical -> dropdown using the choices seen during training
        choices = categories.get(col, [])
        user_input[col] = st.sidebar.selectbox(label, choices, key=f"in_{col}")

# ---- Show the chosen flat ----
st.write("### Your flat")
cols = st.columns(min(3, len(user_input)))
for i, (col, val) in enumerate(user_input.items()):
    label = ft.FEATURE_DEFS[col]["label"]
    cols[i % len(cols)].write(f"**{label}:** {val}")

# ---- Prediction ----
if st.button("Predict resale price", type="primary"):
    row = pd.DataFrame([user_input])
    # Encode categories and line the columns up exactly with the trained model
    row = pd.get_dummies(row, columns=ft.categorical_cols(choice))
    row = row.reindex(columns=prof["columns"], fill_value=0)

    price = prof["model"].predict(row)[0]
    mae = prof["mae"]

    st.success(f"🇸🇬 Estimated resale price: **S${price:,.0f}**")
    st.caption(
        f"Likely range (±1 average error): "
        f"S${max(0, price - mae):,.0f} — S${price + mae:,.0f}  "
        f"·  using **{prof['short']}**"
    )
    st.info(
        "This is an estimate from a teaching model, not a valuation. "
        "Real prices also depend on renovation, exact location, and market timing."
    )

st.divider()

# ---- Overall learning (the story behind the four models) ----
with st.expander("📚 What this app teaches — the full journey & key lessons", expanded=False):
    st.markdown(
        "**The leverage ladder** — how the average error shrank, step by step "
        "(all on unseen test data):"
    )
    st.table(
        pd.DataFrame([
            {"Step": "Baseline (3 features, Linear)", "MAE": "S$102,000", "R²": "~0.50"},
            {"Step": "Enhanced B (+flat_type, town)", "MAE": "S$82,761", "R²": "0.704"},
            {"Step": "Part B+ (+lease, txn_year, model)", "MAE": "S$52,882", "R²": "0.868"},
            {"Step": "Enhanced C (tuned Gradient Boosting)", "MAE": "S$26,439", "R²": "0.964"},
            {"Step": "Enhanced D (stacking ensemble)", "MAE": "S$24,630", "R²": "0.967"},
        ])
    )
    st.markdown(
        """
**🧭 Thinking like a data scientist — five lessons:**

1. **Spend effort where the leverage is.** ~Half the total gain came from *features
   and data cleaning* (S$102k → S$53k); ensembling added only ~6% for ~3× the cost.
   Interrogate the **data** before reaching for fancier models.
2. **A single metric is a trap.** Triangulate **MAE** (dollar intuition) + **MAPE**
   (scale-fair) + **R²** (vs. a mean-guess), and pair the *test* score with the
   *train↔test gap* — Random Forest's lowest MAE hid heavy overfitting.
3. **"Best" is a business decision.** Stacking won on accuracy but costs ~3× and is
   harder to serve & maintain. Choose on the **Pareto frontier of accuracy vs.
   cost / complexity / risk** — often the tuned single model wins.
4. **Trust comes from honest evaluation.** Score on held-out data and show an
   uncertainty band (±MAE), not a false-precision point estimate.
5. **Know when to stop.** When the curve flattens, ROI shifts to *better data*,
   *drift monitoring*, and *validating business impact* — not more modelling.

> **A data scientist's job isn't to maximise a metric — it's to make a
> trustworthy, cost-aware decision under uncertainty.**
"""
    )

st.caption("Module 3 · Machine Learning & GenAI · L06 coaching project")
