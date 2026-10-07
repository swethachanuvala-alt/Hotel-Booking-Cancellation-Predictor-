# 🏨 Aurum Suites – Hotel Booking Cancellation Predictor

A multipage Streamlit app (yellow hotel theme) that predicts whether a hotel
reservation will be cancelled, using the Random Forest model from the
`GGST_11.ipynb` notebook.

**Pages:** Lobby (home) · Booking Desk (single prediction) · Group Bookings (CSV upload) ·
Analytics Suite (trends, model results, feature importance) · About

## Project structure

```
hotel_booking_app/
├── app.py                  # entry point + navigation
├── ui.py                   # theme, CSS, shared helpers
├── model_utils.py          # pipeline, loading, prediction
├── train_model.py          # optional: re-train and re-save the model
├── requirements.txt
├── .streamlit/config.toml  # yellow theme
├── views/                  # the 5 pages
├── model/                  # random_forest_model.pkl, scaler.pkl, label_encoders.pkl
├── data/hotel_booking_cancellation.csv
└── notebook/GGST_11.ipynb
```

## Run locally (Windows CMD, inside VS Code terminal)

```
cd hotel_booking_app
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501

## Push to GitHub

```
git init
git add .
git commit -m "Hotel booking cancellation app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and pick your repository and the `main` branch.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**.

## Using your own notebook model files

The notebook saves `random_forest_model.pkl`, `scaler.pkl` and `label_encoders.pkl`.
To use them, copy those three files into the `model/` folder (replace the existing ones).
If a model file is missing or cannot be loaded, the app automatically re-trains the same
pipeline from `data/hotel_booking_cancellation.csv`, so it never crashes.

## Notes

- Target encoding: 0 = Canceled, 1 = Not_Canceled. The app shows the probability of **cancellation**.
- Risk levels: Low < 33%, Medium 33–66%, High > 66%.
