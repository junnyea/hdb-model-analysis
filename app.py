# =============================================================
# L06 — HDB Resale Price Predictor (deployable Streamlit app)
#
# The input form is built AUTOMATICALLY from features.py — add a feature
# there and a matching slider/dropdown appears here with no other changes.
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


@st.cache_resource(show_spinner="Training the model (first run only)…")
def get_model():
    """Load the saved model, or train one on first run. Cached across reruns."""
    return load_or_train()


bundle = get_model()
model = bundle["model"]
model_columns = bundle["columns"]

# ---- Header ----
st.title("🏡 Singapore HDB Resale Price Predictor")
st.caption(
    f"Powered by a **{bundle['model_name']}** model trained on "
    f"{bundle['n_rows']:,} real resale transactions (2017 onwards)."
)

# ---- Model report card ----
with st.expander("📊 How accurate is this model?", expanded=False):
    st.metric("Average error (MAE)", f"S${bundle['mae']:,.0f}")
    st.write(
        "On average, the prediction is off by this much. We tested it on flats "
        "the model had never seen, so this is an honest estimate."
    )
    st.write("**Models compared during training:**")
    st.table(
        pd.DataFrame(
            [{"Model": n, "Average error (MAE)": f"S${m:,.0f}"}
             for n, m in bundle["all_scores"].items()]
        )
    )

# ---- Inputs (built automatically from features.py) ----
st.sidebar.header("Flat details")
user_input = {}
for f in ft.FEATURES:
    col, label = f["col"], f["label"]
    if f["type"] == "numeric":
        user_input[col] = st.sidebar.slider(
            label, f["min"], f["max"], f["default"], f["step"]
        )
    else:  # categorical -> dropdown using the choices seen during training
        choices = bundle["categories"].get(col, [])
        user_input[col] = st.sidebar.selectbox(label, choices)

# ---- Show the chosen flat ----
st.write("### Your flat")
cols = st.columns(min(3, len(user_input)))
for i, (col, val) in enumerate(user_input.items()):
    label = next(f["label"] for f in ft.FEATURES if f["col"] == col)
    cols[i % len(cols)].write(f"**{label}:** {val}")

# ---- Prediction ----
if st.button("Predict resale price", type="primary"):
    row = pd.DataFrame([user_input])
    # Encode categories and line the columns up exactly with the trained model
    row = pd.get_dummies(row, columns=ft.categorical_cols())
    row = row.reindex(columns=model_columns, fill_value=0)

    price = model.predict(row)[0]
    mae = bundle["mae"]

    st.success(f"🇸🇬 Estimated resale price: **S${price:,.0f}**")
    st.caption(
        f"Likely range (±1 average error): "
        f"S${max(0, price - mae):,.0f} — S${price + mae:,.0f}"
    )
    st.info(
        "This is an estimate from a teaching model, not a valuation. "
        "Real prices also depend on renovation, exact location, and market timing."
    )

st.divider()
st.caption("Module 3 · Machine Learning & GenAI · L06 coaching project")
