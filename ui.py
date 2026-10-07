"""Shared look & feel (yellow hotel theme), HTML helpers and cached loaders."""
from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

import model_utils as mu

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
GOLD = "#F5B800"
GOLD_DEEP = "#C98F00"
GOLD_SOFT = "#FFE9A8"
CREAM = "#FFFBEB"
ESPRESSO = "#2A1F00"
GREEN = "#2E9E5B"
AMBER = "#F08A00"
RED = "#D64545"

BRAND = "Aurum Suites"


def clean(markup: str) -> str:
    """Strip indentation / blank lines so Markdown never treats HTML as code."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def render(markup: str) -> None:
    st.markdown(clean(markup), unsafe_allow_html=True)


def esc(value) -> str:
    return _html.escape(str(value))


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Poppins:wght@300;400;500;600&display=swap');

:root {{
  --gold: {GOLD};
  --gold-deep: {GOLD_DEEP};
  --gold-soft: {GOLD_SOFT};
  --cream: {CREAM};
  --espresso: {ESPRESSO};
}}

html, body, [class*="css"], .stApp {{
  font-family: 'Poppins', 'Segoe UI', sans-serif;
  color: var(--espresso);
}}
.stApp {{
  background:
    radial-gradient(1200px 500px at 85% -10%, #FFF0B8 0%, rgba(255,240,184,0) 60%),
    radial-gradient(900px 400px at -10% 110%, #FFE8A0 0%, rgba(255,232,160,0) 55%),
    var(--cream);
}}
h1, h2, h3, h4 {{
  font-family: 'Playfair Display', Georgia, serif !important;
  color: var(--espresso);
  letter-spacing: .2px;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; padding-bottom: 4rem; max-width: 1180px; }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #2A1F00 0%, #3A2B00 100%);
  border-right: 3px solid var(--gold);
}}
[data-testid="stSidebar"] * {{ color: #FFEFC2; }}
[data-testid="stSidebar"] a {{ border-radius: 10px; }}
[data-testid="stSidebar"] a[aria-current="page"] {{
  background: rgba(245,184,0,.22) !important;
}}
[data-testid="stSidebar"] a:hover {{ background: rgba(245,184,0,.14) !important; }}
.brand {{
  text-align: center; padding: 10px 6px 14px 6px;
  border-bottom: 1px dashed rgba(245,184,0,.45); margin-bottom: 10px;
}}
.brand .logo {{ font-size: 2.1rem; line-height: 1; }}
.brand .name {{
  font-family: 'Playfair Display', serif; font-weight: 800; font-size: 1.35rem;
  color: var(--gold) !important; margin-top: 4px; letter-spacing: 1px;
}}
.brand .tag {{ font-size: .68rem; letter-spacing: 3px; text-transform: uppercase; opacity: .75; }}
.side-note {{
  font-size: .75rem; opacity: .75; border-top: 1px dashed rgba(245,184,0,.35);
  padding-top: 10px; margin-top: 14px; line-height: 1.5;
}}

/* ---------- hero ---------- */
.hero {{
  position: relative; overflow: hidden; border-radius: 28px; padding: 54px 48px;
  background: linear-gradient(120deg, #FFD43B 0%, #F5B800 45%, #E19A00 100%);
  box-shadow: 0 18px 40px rgba(201,143,0,.28);
  margin-bottom: 26px;
}}
.hero::before {{
  content: ""; position: absolute; right: -80px; top: -80px; width: 340px; height: 340px;
  border-radius: 50%; background: rgba(255,255,255,.22);
}}
.hero::after {{
  content: "🏨"; position: absolute; right: 48px; bottom: 6px; font-size: 7.5rem;
  opacity: .9; filter: drop-shadow(0 8px 12px rgba(0,0,0,.18));
}}
.hero .stars {{ letter-spacing: 6px; font-size: 1rem; color: var(--espresso); opacity: .85; }}
.hero h1 {{
  font-size: 2.9rem; line-height: 1.12; margin: 8px 0 12px 0; max-width: 680px;
  color: var(--espresso) !important;
}}
.hero p {{ max-width: 600px; font-size: 1.05rem; color: #4A3700; margin-bottom: 0; }}
.pill {{
  display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: .78rem;
  font-weight: 600; letter-spacing: .6px; background: var(--espresso); color: var(--gold);
}}

/* ---------- cards ---------- */
.card {{
  background: #fff; border-radius: 22px; padding: 22px 24px;
  border: 1px solid #F3DE9A; box-shadow: 0 8px 22px rgba(201,143,0,.10);
  height: 100%;
}}
.card h4 {{ margin: 0 0 6px 0; font-size: 1.15rem; }}
.card p {{ margin: 0; color: #5B4709; font-size: .92rem; line-height: 1.55; }}
.card .icon {{ font-size: 1.8rem; margin-bottom: 6px; }}

.kpi {{
  background: #fff; border-radius: 20px; padding: 18px 20px; text-align: left;
  border: 1px solid #F3DE9A; border-left: 8px solid var(--gold);
  box-shadow: 0 8px 22px rgba(201,143,0,.10);
}}
.kpi .v {{ font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 800; line-height: 1.1; }}
.kpi .l {{ font-size: .78rem; letter-spacing: 1.5px; text-transform: uppercase; color: #8A6A00; margin-top: 4px; }}

.step {{
  background: linear-gradient(160deg, #fff 0%, #FFF6D6 100%); border-radius: 22px;
  padding: 22px; border: 1px solid #F3DE9A; height: 100%; position: relative;
}}
.step .n {{
  width: 38px; height: 38px; border-radius: 50%; background: var(--espresso); color: var(--gold);
  display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 10px;
  font-family: 'Playfair Display', serif;
}}
.step h4 {{ margin: 0 0 4px 0; }}
.step p {{ margin: 0; font-size: .9rem; color: #5B4709; }}

.section-title {{
  display: flex; align-items: center; gap: 12px; margin: 30px 0 14px 0;
}}
.section-title .bar {{ width: 8px; height: 30px; border-radius: 6px; background: var(--gold); }}
.section-title h2 {{ margin: 0; font-size: 1.7rem; }}
.section-sub {{ color: #6B5410; margin: -6px 0 16px 20px; font-size: .95rem; }}

/* ---------- page banner (inner pages) ---------- */
.banner {{
  border-radius: 24px; padding: 28px 34px; margin-bottom: 22px;
  background: linear-gradient(110deg, #FFD43B, #F5B800 60%, #E8A400);
  box-shadow: 0 12px 28px rgba(201,143,0,.25);
  display: flex; align-items: center; justify-content: space-between; gap: 20px;
}}
.banner h1 {{ margin: 0; font-size: 2.1rem; color: var(--espresso) !important; }}
.banner p {{ margin: 4px 0 0 0; color: #4A3700; }}
.banner .emoji {{ font-size: 3.4rem; }}

/* ---------- key-card result ---------- */
.keycard {{
  position: relative; background: #fff; border-radius: 26px; overflow: hidden;
  border: 2px solid var(--espresso); box-shadow: 0 16px 36px rgba(42,31,0,.18);
  display: flex; flex-wrap: wrap;
}}
.keycard .left {{
  flex: 0 0 280px; background: var(--espresso); color: #FFEFC2; padding: 28px 22px;
  display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
  border-right: 3px dashed var(--gold);
}}
.keycard .right {{ flex: 1 1 320px; padding: 26px 30px; }}
.keycard .notch-t, .keycard .notch-b {{
  position: absolute; left: 280px; width: 26px; height: 26px; border-radius: 50%;
  background: var(--cream); border: 2px solid var(--espresso); transform: translateX(-50%);
}}
.keycard .notch-t {{ top: -14px; }}
.keycard .notch-b {{ bottom: -14px; }}
.gauge {{
  --p: 0; --c: {GOLD};
  width: 170px; height: 170px; border-radius: 50%;
  background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,239,194,.18) 0);
  display: flex; align-items: center; justify-content: center; margin-bottom: 14px;
}}
.gauge .inner {{
  width: 130px; height: 130px; border-radius: 50%; background: var(--espresso);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}}
.gauge .pct {{ font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 800; color: #fff; line-height: 1; }}
.gauge .lbl {{ font-size: .62rem; letter-spacing: 2px; text-transform: uppercase; color: var(--gold); margin-top: 4px; }}
.badge {{
  display: inline-block; padding: 5px 16px; border-radius: 999px; font-weight: 600;
  font-size: .8rem; letter-spacing: 1px; text-transform: uppercase; color: #fff;
}}
.badge.low {{ background: {GREEN}; }}
.badge.medium {{ background: {AMBER}; }}
.badge.high {{ background: {RED}; }}
.keycard h3 {{ margin: 0 0 4px 0; font-size: 1.7rem; }}
.keycard .sub {{ color: #6B5410; margin-bottom: 14px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
.chip {{
  background: #FFF3C4; border: 1px solid #F0D673; border-radius: 12px; padding: 6px 12px;
  font-size: .8rem; color: #5B4709;
}}
.chip b {{ color: var(--espresso); }}

/* ---------- horizontal bars ---------- */
.hbar {{ display: flex; align-items: center; gap: 12px; margin: 9px 0; }}
.hbar .lab {{ flex: 0 0 190px; font-size: .86rem; color: #3E2F00; }}
.hbar .track {{ flex: 1; background: #FFF0BD; border-radius: 999px; height: 18px; overflow: hidden; }}
.hbar .fill {{
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, #FFD43B, #E19A00);
}}
.hbar .fill.dark {{ background: linear-gradient(90deg, #5B4300, #2A1F00); }}
.hbar .val {{ flex: 0 0 64px; text-align: right; font-weight: 600; font-size: .86rem; }}

/* ---------- confusion matrix ---------- */
.cm {{ display: grid; grid-template-columns: 90px 1fr 1fr; gap: 8px; max-width: 460px; }}
.cm div {{ border-radius: 14px; padding: 16px 8px; text-align: center; }}
.cm .h {{ background: transparent; font-size: .75rem; letter-spacing: 1px; text-transform: uppercase; color: #8A6A00; padding: 8px; }}
.cm .good {{ background: var(--gold); color: var(--espresso); font-weight: 700; font-size: 1.3rem; }}
.cm .bad {{ background: #FFF0BD; color: #5B4709; font-size: 1.15rem; }}

/* ---------- misc widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  background: var(--espresso); color: var(--gold); border: 2px solid var(--espresso);
  border-radius: 14px; padding: .6rem 1.4rem; font-weight: 600; letter-spacing: .5px;
  transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
  background: var(--gold); color: var(--espresso); border-color: var(--espresso);
  transform: translateY(-1px);
}}
[data-testid="stForm"] {{
  background: rgba(255,255,255,.75); border: 1px solid #F3DE9A; border-radius: 24px; padding: 22px 26px;
}}
.form-h {{ font-family: 'Playfair Display', serif; font-weight: 700; font-size: 1.15rem; margin: 4px 0 8px 0; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-family: 'Playfair Display', serif; }}
.footer {{ text-align: center; color: #8A6A00; font-size: .8rem; margin-top: 40px; letter-spacing: 1px; }}

@media (max-width: 760px) {{
  .hero {{ padding: 32px 22px; }}
  .hero h1 {{ font-size: 2rem; }}
  .hero::after {{ display: none; }}
  .keycard .left {{ flex-basis: 100%; border-right: none; border-bottom: 3px dashed var(--gold); }}
  .keycard .notch-t, .keycard .notch-b {{ display: none; }}
  .hbar .lab {{ flex-basis: 110px; }}
  .banner .emoji {{ display: none; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(clean(_CSS), unsafe_allow_html=True)


def sidebar_brand() -> None:
    with st.sidebar:
        render(
            f"""
            <div class="brand">
              <div class="logo">🔑</div>
              <div class="name">{BRAND}</div>
              <div class="tag">Cancellation Desk</div>
            </div>
            """
        )


def sidebar_footer() -> None:
    with st.sidebar:
        render(
            """
            <div class="side-note">
              Random Forest model · trained on 36,285 reservations.<br>
              Predictions are decision support, not a guarantee.
            </div>
            """
        )


def banner(title: str, subtitle: str, emoji: str = "🏨") -> None:
    render(
        f"""
        <div class="banner">
          <div><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>
          <div class="emoji">{emoji}</div>
        </div>
        """
    )


def section(title: str, subtitle: str = "") -> None:
    render(
        f"""
        <div class="section-title"><div class="bar"></div><h2>{esc(title)}</h2></div>
        """
    )
    if subtitle:
        render(f'<div class="section-sub">{esc(subtitle)}</div>')


def kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def bars(items, suffix: str = "%", dark: bool = False, max_value: float | None = None) -> None:
    """Horizontal bar list. items = [(label, value)]; value is a number."""
    items = list(items)
    if not items:
        return
    top = max_value if max_value else max(v for _, v in items) or 1
    rows = []
    for label, value in items:
        width = max(2.0, min(100.0, value / top * 100))
        cls = "fill dark" if dark else "fill"
        rows.append(
            f'<div class="hbar"><div class="lab">{esc(label)}</div>'
            f'<div class="track"><div class="{cls}" style="width:{width:.1f}%"></div></div>'
            f'<div class="val">{value:.1f}{suffix}</div></div>'
        )
    render("".join(rows))


def footer() -> None:
    render(f'<div class="footer">© {BRAND} · Hotel Booking Cancellation Predictor · Built with Streamlit</div>')


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Preparing the prediction engine…")
def get_artifacts():
    return mu.load_artifacts()


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    df = mu.load_raw_data()
    df["canceled"] = (df[mu.TARGET] == "Canceled").astype(int)
    df["reservation_date"] = pd.to_datetime(df["date of reservation"], errors="coerce")
    return df
