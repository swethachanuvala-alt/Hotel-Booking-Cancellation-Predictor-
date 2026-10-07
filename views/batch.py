import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import model_utils as mu  # noqa: E402
import ui  # noqa: E402

ui.banner(
    "Group Bookings",
    "Upload a spreadsheet of reservations and rank every booking by cancellation risk.",
    "🧳",
)

model, scaler, encoders, meta, source = ui.get_artifacts()
data = ui.get_data()

NUMERIC = [c for c in mu.FEATURES if c not in mu.CAT_COLS]
ALLOWED = {
    "type of meal": mu.MEAL_OPTIONS,
    "room type": mu.ROOM_OPTIONS,
    "market segment type": mu.SEGMENT_OPTIONS,
}

left, right = st.columns([3, 2])
with left:
    ui.render(
        """
        <div class="card">
          <h4>What your file needs</h4>
          <p>A CSV with these 14 columns (same names as the original dataset):<br>
          <b>number of adults, number of children, number of weekend nights, number of week nights,
          type of meal, car parking space, room type, lead time, market segment type, repeated,
          P-C, P-not-C, average price, special requests</b>.<br>
          A <b>Booking_ID</b> column is optional and will be kept in the results.</p>
        </div>
        """
    )
with right:
    template = data[["Booking_ID"] + mu.FEATURES].head(5)
    st.download_button(
        "Download a sample template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="booking_template.csv",
        mime="text/csv",
    )
    use_demo = st.checkbox("Or try 300 random bookings from the dataset")

st.markdown("&nbsp;", unsafe_allow_html=True)
upload = st.file_uploader("Upload your reservations (CSV)", type=["csv"])

raw = None
if upload is not None:
    try:
        raw = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"That file could not be read as a CSV: {exc}")
elif use_demo:
    raw = data[["Booking_ID"] + mu.FEATURES].sample(300, random_state=7).reset_index(drop=True)

if raw is not None:
    if raw.empty:
        st.error("The file has no rows.")
        st.stop()

    # tolerant column names (case / spacing)
    canon = {c.lower(): c for c in mu.FEATURES + ["Booking_ID"]}
    raw.columns = [canon.get(str(c).strip().lower(), str(c).strip()) for c in raw.columns]

    missing = [c for c in mu.FEATURES if c not in raw.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    work = raw.copy()
    n_in = len(work)

    # numeric columns
    for c in NUMERIC:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    bad_numeric = work[NUMERIC].isna().any(axis=1)

    # categorical columns
    for c in mu.CAT_COLS:
        work[c] = work[c].astype(str).str.strip()
    bad_cat = pd.Series(False, index=work.index)
    for c, allowed in ALLOWED.items():
        invalid = ~work[c].isin(allowed)
        if invalid.any():
            shown = ", ".join(sorted(work.loc[invalid, c].unique())[:5])
            st.warning(f"Column '{c}' has unknown values ({shown}). Allowed: {', '.join(allowed)}. "
                       f"{int(invalid.sum())} row(s) skipped.")
        bad_cat |= invalid

    skip = bad_numeric | bad_cat
    if bad_numeric.any():
        st.warning(f"{int(bad_numeric.sum())} row(s) have missing or non-numeric values and were skipped.")
    work = work.loc[~skip].copy()

    if work.empty:
        st.error("No valid rows to score after checking the file.")
        st.stop()

    try:
        prob = mu.predict_cancel_proba(model, scaler, encoders, work)
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
        st.stop()

    out = work.copy()
    out["Cancel probability %"] = (prob * 100).round(1)
    out["Risk level"] = [mu.risk_level(float(p)) for p in prob]
    out["Prediction"] = ["Likely to cancel" if p >= 0.5 else "Likely to arrive" for p in prob]
    out = out.sort_values("Cancel probability %", ascending=False).reset_index(drop=True)

    n = len(out)
    n_cancel = int((out["Prediction"] == "Likely to cancel").sum())
    n_high = int((out["Risk level"] == "High").sum())
    avg_risk = float(out["Cancel probability %"].mean())

    ui.section("Results", f"{n:,} of {n_in:,} bookings scored")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(ui.clean(ui.kpi(f"{n:,}", "Bookings scored")), unsafe_allow_html=True)
    k2.markdown(ui.clean(ui.kpi(f"{n_cancel:,}", "Likely to cancel")), unsafe_allow_html=True)
    k3.markdown(ui.clean(ui.kpi(f"{n_high:,}", "High-risk bookings")), unsafe_allow_html=True)
    k4.markdown(ui.clean(ui.kpi(f"{avg_risk:.1f}%", "Average risk")), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    dist = out["Risk level"].value_counts()
    ui.bars(
        [(f"{lvl} risk", float(dist.get(lvl, 0)) / n * 100) for lvl in ["High", "Medium", "Low"]],
        suffix="%",
        max_value=100,
    )

    lead = ["Booking_ID"] if "Booking_ID" in out.columns else []
    cols = lead + ["Cancel probability %", "Risk level", "Prediction"] + [
        c for c in mu.FEATURES if c not in lead
    ]
    st.dataframe(out[cols], hide_index=True)

    st.download_button(
        "Download ranked results (CSV)",
        out[cols].to_csv(index=False).encode("utf-8"),
        file_name="cancellation_risk_results.csv",
        mime="text/csv",
    )

ui.footer()
