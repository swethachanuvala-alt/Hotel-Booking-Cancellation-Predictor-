import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import model_utils as mu  # noqa: E402
import ui  # noqa: E402

ui.banner("Booking Desk", "Enter a reservation and see how likely it is to be cancelled.", "🛎️")

model, scaler, encoders, meta, source = ui.get_artifacts()
data = ui.get_data()

# --------------------------------------------------------------------------- #
# Form
# --------------------------------------------------------------------------- #
with st.form("booking_form"):
    st.markdown('<div class="form-h">👤 Guests</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    adults = g1.number_input("Adults", min_value=0, max_value=4, value=2, step=1)
    children = g2.number_input("Children", min_value=0, max_value=10, value=0, step=1)
    repeated = g3.selectbox("Returning guest?", ["No", "Yes"])

    h1, h2 = st.columns(2)
    prev_canceled = h1.number_input(
        "Previous cancellations by this guest", min_value=0, max_value=13, value=0, step=1
    )
    prev_kept = h2.number_input(
        "Previous bookings NOT cancelled", min_value=0, max_value=58, value=0, step=1
    )

    st.markdown('<div class="form-h">🌙 The stay</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    weekend_nights = s1.number_input("Weekend nights", min_value=0, max_value=7, value=1, step=1)
    week_nights = s2.number_input("Week nights", min_value=0, max_value=17, value=2, step=1)
    lead_time = s3.number_input(
        "Lead time (days before arrival)", min_value=0, max_value=443, value=60, step=1
    )

    st.markdown('<div class="form-h">🛏️ Room, meals & price</div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    room = r1.selectbox("Room type", mu.ROOM_OPTIONS)
    meal = r2.selectbox("Meal plan", mu.MEAL_OPTIONS)
    price = r3.number_input(
        "Average price per night", min_value=0.0, max_value=540.0, value=100.0, step=1.0
    )

    p1, p2, p3 = st.columns(3)
    segment = p1.selectbox("Booking channel", mu.SEGMENT_OPTIONS)
    requests = p2.slider("Special requests", min_value=0, max_value=5, value=0)
    parking = p3.selectbox("Car parking space", ["No", "Yes"])

    submitted = st.form_submit_button("Predict cancellation risk")

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
if submitted:
    row = pd.DataFrame(
        [
            {
                "number of adults": int(adults),
                "number of children": int(children),
                "number of weekend nights": int(weekend_nights),
                "number of week nights": int(week_nights),
                "type of meal": meal,
                "car parking space": 1 if parking == "Yes" else 0,
                "room type": room,
                "lead time": int(lead_time),
                "market segment type": segment,
                "repeated": 1 if repeated == "Yes" else 0,
                "P-C": int(prev_canceled),
                "P-not-C": int(prev_kept),
                "average price": float(price),
                "special requests": int(requests),
            }
        ]
    )
    try:
        p = float(mu.predict_cancel_proba(model, scaler, encoders, row)[0])
        st.session_state["result"] = {
            "p": p,
            "nights": int(weekend_nights) + int(week_nights),
            "guests": int(adults) + int(children),
            "lead": int(lead_time),
            "segment": segment,
            "requests": int(requests),
            "room": room,
            "price": float(price),
        }
    except Exception as exc:  # pragma: no cover
        st.session_state.pop("result", None)
        st.error(f"Sorry, the prediction could not be completed: {exc}")

res = st.session_state.get("result")

if res:
    if res["nights"] == 0:
        st.warning("This booking has 0 nights. Please double-check the weekend and week nights.")
    if res["guests"] == 0:
        st.warning("This booking has 0 guests. Please double-check adults and children.")

    p = res["p"]
    level = mu.risk_level(p)
    color = {"Low": ui.GREEN, "Medium": ui.AMBER, "High": ui.RED}[level]
    verdict = "Likely to cancel" if p >= 0.5 else "Likely to arrive"
    blurb = {
        "Low": "This guest looks committed. A smooth, standard stay is expected.",
        "Medium": "Could go either way. A small nudge before arrival can tip it in your favour.",
        "High": "This reservation is at real risk of falling through. Act early.",
    }[level]

    lead_label = mu.lead_bucket([res["lead"]]).iloc[0]
    lead_rate = data.loc[mu.lead_bucket(data["lead time"]).to_numpy() == lead_label, "canceled"].mean() * 100
    seg_rate = data.loc[data["market segment type"] == res["segment"], "canceled"].mean() * 100
    req_rate = data.loc[data["special requests"] == res["requests"], "canceled"].mean() * 100

    st.markdown("&nbsp;", unsafe_allow_html=True)
    ui.render(
        f"""
        <div class="keycard">
          <div class="notch-t"></div><div class="notch-b"></div>
          <div class="left">
            <div class="gauge" style="--p:{p * 100:.0f};--c:{color}">
              <div class="inner"><div class="pct">{p * 100:.0f}%</div><div class="lbl">cancel risk</div></div>
            </div>
            <span class="badge {level.lower()}">{level} risk</span>
          </div>
          <div class="right">
            <h3>{verdict}</h3>
            <div class="sub">{blurb}</div>
            <div class="chips">
              <div class="chip">Stay <b>{res['nights']} night(s)</b></div>
              <div class="chip">Guests <b>{res['guests']}</b></div>
              <div class="chip">Lead time <b>{res['lead']} days</b></div>
              <div class="chip">Room <b>{ui.esc(res['room'])}</b></div>
              <div class="chip">Channel <b>{ui.esc(res['segment'])}</b></div>
              <div class="chip">Rate <b>{res['price']:.0f} / night</b></div>
            </div>
            <div class="chips" style="margin-top:16px">
              <div class="chip">Similar lead time ({lead_label}) cancelled <b>{lead_rate:.0f}%</b> of the time</div>
              <div class="chip">{ui.esc(res['segment'])} channel cancelled <b>{seg_rate:.0f}%</b></div>
              <div class="chip">{res['requests']} special request(s) cancelled <b>{req_rate:.0f}%</b></div>
            </div>
          </div>
        </div>
        """
    )

    ui.section("Concierge recommendations")
    tips = {
        "High": [
            ("💳", "Secure the booking", "Ask for a card guarantee or a small deposit before the stay."),
            ("📧", "Reconfirm early", "Send a personal reconfirmation about a week before arrival."),
            ("🎁", "Give a reason to stay", "Offer a perk such as breakfast or a late checkout for keeping the booking."),
        ],
        "Medium": [
            ("📧", "Friendly reminder", "Send a pre-arrival message a few days before check-in."),
            ("🍽️", "Add something special", "Suggest a meal plan or an experience to raise commitment."),
            ("📋", "Keep a waitlist", "Have a backup guest ready in case the room is released."),
        ],
        "Low": [
            ("✅", "Standard process", "No extra action needed. Treat it as a confirmed stay."),
            ("👋", "Warm welcome", "A short welcome message before arrival keeps the guest engaged."),
            ("⬆️", "Upsell", "A good moment to offer an upgrade or add-on services."),
        ],
    }[level]
    cols = st.columns(3)
    for col, (icon, title, text) in zip(cols, tips):
        col.markdown(
            ui.clean(f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>'),
            unsafe_allow_html=True,
        )
    st.caption(
        "The percentages in the chips come from the historical data. "
        "The risk score comes from the Random Forest model."
    )
else:
    st.info("Fill in the reservation above and press **Predict cancellation risk**.")

if source == "retrained":
    st.caption("Model files were rebuilt from the bundled dataset for this session.")

ui.footer()
