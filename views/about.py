import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import ui  # noqa: E402

ui.banner("About this project", "A machine learning model that predicts hotel booking cancellations.", "📖")

ui.section("The problem")
ui.render(
    """
    <div class="card"><p>
    Cancellations leave rooms empty and revenue uncertain. This project learns from 36,285 past
    reservations to estimate the chance that a new booking will be cancelled, so a hotel can
    reconfirm, secure or resell early.
    </p></div>
    """
)

ui.section("How the model was built")
steps = [
    ("1", "Clean", "Removed Booking_ID and the reservation date; no missing values in the data."),
    ("2", "Encode", "Label-encoded meal plan, room type, booking channel and the target."),
    ("3", "Split & scale", "Stratified 80/20 train-test split, then standard scaling fitted on the training set."),
    ("4", "Balance", "SMOTE added synthetic cancelled bookings to the training set only."),
    ("5", "Tune & choose", "Compared 5 models with randomised search and cross-validation. Random Forest won."),
]
cols = st.columns(5)
for col, (n, t, d) in zip(cols, steps):
    col.markdown(
        ui.clean(f'<div class="step"><div class="n">{n}</div><h4>{t}</h4><p>{d}</p></div>'),
        unsafe_allow_html=True,
    )

ui.section("Data dictionary")
dictionary = pd.DataFrame(
    [
        ("number of adults / children", "Guests on the booking"),
        ("number of weekend / week nights", "Length of stay, split by night type"),
        ("type of meal", "Meal plan chosen (Meal Plan 1-3 or Not Selected)"),
        ("car parking space", "1 if a parking space was requested"),
        ("room type", "Room type reserved (Room_Type 1-7)"),
        ("lead time", "Days between booking and arrival"),
        ("market segment type", "Booking channel: Online, Offline, Corporate, Complementary, Aviation"),
        ("repeated", "1 if the guest has stayed before"),
        ("P-C / P-not-C", "Guest's previous cancelled / not cancelled bookings"),
        ("average price", "Average price per night"),
        ("special requests", "Number of special requests made"),
        ("booking status", "Target: Canceled or Not_Canceled"),
    ],
    columns=["Column", "Meaning"],
)
st.dataframe(dictionary, hide_index=True)

ui.section("Good to know")
ui.render(
    """
    <div class="card"><p>
    The score is a probability based on past patterns, not a certainty. Use it to prioritise
    follow-ups, not to refuse guests. Hotels, seasons and customer behaviour change, so the model
    should be retrained on fresh data from time to time (run <b>python train_model.py</b>).
    </p></div>
    """
)

ui.footer()
