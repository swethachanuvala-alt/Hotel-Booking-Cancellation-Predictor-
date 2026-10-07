import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import model_utils as mu  # noqa: E402
import ui  # noqa: E402

ui.banner(
    "Analytics Suite",
    "What the reservation history says about cancellations, and how well the model reads it.",
    "📊",
)

model, scaler, encoders, meta, source = ui.get_artifacts()
df = ui.get_data()

PRETTY = {
    "number of adults": "Adults",
    "number of children": "Children",
    "number of weekend nights": "Weekend nights",
    "number of week nights": "Week nights",
    "type of meal": "Meal plan",
    "car parking space": "Car parking",
    "room type": "Room type",
    "lead time": "Lead time",
    "market segment type": "Booking channel",
    "repeated": "Returning guest",
    "P-C": "Past cancellations",
    "P-not-C": "Past stays kept",
    "average price": "Average price",
    "special requests": "Special requests",
}


def rate_by(series: pd.Series, order=None):
    g = df.groupby(series.to_numpy())["canceled"].mean() * 100
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    else:
        g = g.sort_values(ascending=False)
    return [(str(k), float(v)) for k, v in g.items()]


tab1, tab2, tab3 = st.tabs(["Booking trends", "Model performance", "What drives cancellations"])

# --------------------------------------------------------------------------- #
with tab1:
    overall = df["canceled"].mean() * 100
    ui.section("Cancellation rate by segment", f"Share of reservations cancelled. Overall rate: {overall:.1f}%")

    a, b = st.columns(2)
    with a:
        st.markdown("**Booking channel**")
        ui.bars(rate_by(df["market segment type"]), max_value=100)
        st.markdown("**Lead time**")
        ui.bars(rate_by(mu.lead_bucket(df["lead time"]), order=mu.LEAD_LABELS), max_value=100)
    with b:
        st.markdown("**Special requests made**")
        ui.bars(
            rate_by(df["special requests"].astype(int).astype(str), order=[str(i) for i in range(6)]),
            max_value=100,
        )
        st.markdown("**Room type**")
        ui.bars(rate_by(df["room type"], order=mu.ROOM_OPTIONS), max_value=100)

    c, d = st.columns(2)
    with c:
        st.markdown("**Meal plan**")
        ui.bars(rate_by(df["type of meal"], order=mu.MEAL_OPTIONS), max_value=100)
    with d:
        months = df["reservation_date"].dt.month
        valid = months.notna()
        if valid.any():
            names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            g = df.loc[valid].groupby(months[valid].astype(int).to_numpy())["canceled"].mean() * 100
            st.markdown("**Reservation month**")
            ui.bars([(names[int(m) - 1], float(v)) for m, v in g.items()], max_value=100)

    st.caption("Channels, rooms and meal plans with very few bookings (for example Meal Plan 3, Room_Type 3) "
               "give unstable percentages.")

# --------------------------------------------------------------------------- #
with tab2:
    ui.section("Model comparison", "Tuned models, evaluated on the 20% hold-out test set from the notebook")
    results = pd.DataFrame(
        [
            ("Random Forest", 0.8871, 0.9198, 0.9117, 0.9157, 0.9492),
            ("XGBoost", 0.8827, 0.9189, 0.9055, 0.9122, 0.9482),
            ("Decision Tree", 0.8621, 0.9175, 0.8733, 0.8949, 0.8821),
            ("KNN", 0.8611, 0.9156, 0.8739, 0.8943, 0.9210),
            ("Logistic Regression", 0.7775, 0.8813, 0.7731, 0.8237, 0.8605),
        ],
        columns=["Model", "Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
    )
    ui.bars([(r.Model, r.Accuracy * 100) for r in results.itertuples()], max_value=100)
    st.dataframe(results, hide_index=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        ui.section("Random Forest confusion matrix", "Test set, 7,257 reservations")
        ui.render(
            """
            <div class="cm">
              <div class="h"></div><div class="h">Predicted<br>cancelled</div><div class="h">Predicted<br>arrived</div>
              <div class="h">Actually<br>cancelled</div><div class="good">1,990</div><div class="bad">388</div>
              <div class="h">Actually<br>arrived</div><div class="bad">431</div><div class="good">4,448</div>
            </div>
            """
        )
    with right:
        ui.section("Why Random Forest", "The selected model")
        ui.render(
            """
            <div class="card"><p>
            Random Forest scored the highest accuracy (88.7%) and the highest ROC-AUC (0.949).
            Five-fold cross-validation on the training data gave a similar 88.3% with a very small
            spread, which suggests the result is stable. About 8 in 10 real cancellations are caught,
            and when the model says a guest will arrive it is right about 92% of the time.
            </p></div>
            """
        )

# --------------------------------------------------------------------------- #
with tab3:
    ui.section("Feature importance", "Which booking details the Random Forest relies on most")
    try:
        imp = pd.Series(model.feature_importances_, index=mu.FEATURES).sort_values(ascending=False)
        ui.bars([(PRETTY[k], float(v) * 100) for k, v in imp.items()], dark=True)
        top = imp.index[0]
        st.caption(
            f"'{PRETTY[top]}' is the strongest signal. Importance shows how much the model uses a feature, "
            "not that it causes cancellations."
        )
    except Exception:
        st.info("Feature importance is not available for the loaded model.")

    ui.section("A simple reading of the data")
    canceled_lead = df.loc[df["canceled"] == 1, "lead time"].mean()
    kept_lead = df.loc[df["canceled"] == 0, "lead time"].mean()
    k1, k2, k3 = st.columns(3)
    k1.markdown(ui.clean(ui.kpi(f"{canceled_lead:.0f} days", "Avg lead time, cancelled")), unsafe_allow_html=True)
    k2.markdown(ui.clean(ui.kpi(f"{kept_lead:.0f} days", "Avg lead time, kept")), unsafe_allow_html=True)
    price_c = df.loc[df["canceled"] == 1, "average price"].mean()
    price_k = df.loc[df["canceled"] == 0, "average price"].mean()
    k3.markdown(
        ui.clean(ui.kpi(f"{price_c:.0f} vs {price_k:.0f}", "Avg price, cancelled vs kept")),
        unsafe_allow_html=True,
    )

ui.footer()
