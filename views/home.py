import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

import ui  # noqa: E402

df = ui.get_data()

total = len(df)
cancel_rate = df["canceled"].mean() * 100
avg_lead = df["lead time"].mean()
avg_price = df["average price"].mean()

ui.render(
    """
    <div class="hero">
      <div class="stars">★ ★ ★ ★ ★</div>
      <h1>Know who will cancel before they ever check in.</h1>
      <p>Welcome to the Cancellation Desk. Our Random Forest concierge reads every reservation
      and tells you how likely it is to be cancelled, so rooms stay full and revenue stays safe.</p>
      <div style="margin-top:20px"><span class="pill">88.7% ACCURACY · RANDOM FOREST</span></div>
    </div>
    """
)

c1, c2, _ = st.columns([1.2, 1.2, 3])
with c1:
    st.page_link("views/predict.py", label="Check a booking", icon="🛎️")
with c2:
    st.page_link("views/batch.py", label="Upload many", icon="🧳")

ui.section("Today at the property", "Numbers from the reservation history behind the model")
k1, k2, k3, k4 = st.columns(4)
k1.markdown(ui.clean(ui.kpi(f"{total:,}", "Reservations studied")), unsafe_allow_html=True)
k2.markdown(ui.clean(ui.kpi(f"{cancel_rate:.1f}%", "Were cancelled")), unsafe_allow_html=True)
k3.markdown(ui.clean(ui.kpi(f"{avg_lead:.0f} days", "Average lead time")), unsafe_allow_html=True)
k4.markdown(ui.clean(ui.kpi(f"{avg_price:.0f}", "Average nightly rate")), unsafe_allow_html=True)

ui.section("How the concierge works")
s1, s2, s3 = st.columns(3)
steps = [
    ("1", "Enter the booking", "Guests, nights, room, meal plan, lead time and price. Takes under a minute."),
    ("2", "The model reads it", "A Random Forest of 200 trees, trained on 36,285 past reservations, scores the booking."),
    ("3", "Act on the risk", "Get a cancellation probability, a risk level and practical next steps for the front desk."),
]
for col, (n, title, text) in zip((s1, s2, s3), steps):
    col.markdown(
        ui.clean(f'<div class="step"><div class="n">{n}</div><h4>{title}</h4><p>{text}</p></div>'),
        unsafe_allow_html=True,
    )

ui.section("Everything in one hotel")
f1, f2, f3, f4 = st.columns(4)
feats = [
    ("🛎️", "Booking Desk", "Score a single reservation and see the risk on a key card."),
    ("🧳", "Group Bookings", "Upload a CSV, score every row and download the ranked list."),
    ("📊", "Analytics Suite", "See what drives cancellations: lead time, price, channel and more."),
    ("📖", "About", "Data, method and model details in plain language."),
]
for col, (icon, title, text) in zip((f1, f2, f3, f4), feats):
    col.markdown(
        ui.clean(f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>'),
        unsafe_allow_html=True,
    )

ui.footer()
