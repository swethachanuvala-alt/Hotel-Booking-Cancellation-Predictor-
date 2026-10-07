"""Entry point: python -m streamlit run app.py"""
import streamlit as st

st.set_page_config(
    page_title="Aurum Suites | Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui import inject_css, sidebar_brand, sidebar_footer  # noqa: E402

inject_css()

pages = [
    st.Page("views/home.py", title="Lobby", icon="🏨", default=True),
    st.Page("views/predict.py", title="Booking Desk", icon="🛎️"),
    st.Page("views/batch.py", title="Group Bookings", icon="🧳"),
    st.Page("views/insights.py", title="Analytics Suite", icon="📊"),
    st.Page("views/about.py", title="About", icon="📖"),
]

nav = st.navigation(pages)
sidebar_brand()
sidebar_footer()
nav.run()
