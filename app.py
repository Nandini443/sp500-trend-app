# app.py  (FULL UPDATED COLORFUL VERSION)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="S&P 500 Trend Prediction", layout="wide")

# -----------------------------
# 🎨 CSS THEME (Colorful)
# -----------------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #0b1220 100%);
    color: #e5e7eb;
}

/* Title */
h1 {
    color: #ffffff !important;
    font-weight: 900 !important;
    letter-spacing: 0.5px;
}

/* Subheaders */
h2, h3 {
    color: #e0e7ff !important;
    font-weight: 800 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Dataframe card */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

/* Metric label */
div[data-testid="stMetricLabel"] p {
    color: #c7d2fe !important;
    font-weight: 700 !important;
}

/* Metric value */
div[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 900 !important;
}

/* Number input / text input style */
div[data-baseweb="input"] {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
}

/* Top padding */
.block-container {
    padding-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.title("📈 S&P 500 Trend Prediction (LSTM)")
st.caption("Dataset: sp 500.csv | Model: LSTM | Output: Next period UP/DOWN")

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "sp500_lstm_model.keras"
SCALER_PATH = "sp500_scaler.pkl"
CSV_PATH = "sp 500.csv"

# -----------------------------
# Load model + scaler (safe)
# -----------------------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"❌ Scaler load failed: {e}")
    st.stop()

# -----------------------------
# Load & clean dataset
# -----------------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Date parse + sort
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Convert numeric columns (remove commas)
    num_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for c in num_cols:
        df[c] = (
            df[c].astype(str)
                 .str.replace(",", "", regex=False)
                 .str.strip()
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    return df

try:
    df = load_data(CSV_PATH)
except Exception as e:
    st.error(f"❌ CSV load failed: {e}")
    st.stop()

# -----------------------------
# Indicators (same as training)
# -----------------------------
df["SMA3"] = df["Close"].rolling(3).mean()
df["SMA6"] = df["Close"].rolling(6).mean()

delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(6).mean()
loss = (-delta.clip(upper=0)).rolling(6).mean()
rs = gain / (loss + 1e-9)
df["RSI6"] = 100 - (100 / (1 + rs))

df = df.dropna().reset_index(drop=True)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Controls")
LOOKBACK = st.sidebar.number_input(
    "Lookback (months)",
    min_value=3,
    max_value=24,
    value=6,
    step=1
)

# -----------------------------
# Layout
# -----------------------------
colA, colB = st.columns([1.1, 1])

with colA:
    st.subheader("📊 Data Preview (last 10 rows)")
    st.dataframe(df.tail(10), use_container_width=True)

with colB:
    st.subheader("📉 Close Price Chart")
    st.line_chart(df.set_index("Date")["Close"])

# -----------------------------
# Prediction
# -----------------------------
st.subheader("🔮 Next Period Trend Prediction")

if len(df) < LOOKBACK:
    st.error(f"Not enough rows for lookback={LOOKBACK}. Need at least {LOOKBACK} rows.")
else:
    features = df[["Close", "SMA3", "SMA6", "RSI6"]].copy()

    # Must match training scaler
    scaled = scaler.transform(features)

    last_seq = scaled[-LOOKBACK:, :]          # (LOOKBACK, 4)
    X_input = np.expand_dims(last_seq, axis=0) # (1, LOOKBACK, 4)

    prob_up = float(model.predict(X_input, verbose=0).ravel()[0])
    pred = 1 if prob_up >= 0.5 else 0

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Probability UP", f"{prob_up * 100:.2f}%")

    with c2:
        # Colorful badge
        badge = "UP 📈" if pred == 1 else "DOWN 📉"
        color = "#22c55e" if pred == 1 else "#ef4444"
        st.markdown(
            f"""
            <div style="
                display:inline-block;
                padding:12px 18px;
                border-radius:999px;
                background:{color};
                color:#0b1220;
                font-weight:900;
                font-size:18px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.30);
            ">
                {badge}
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.metric("Last Close", f"{df['Close'].iloc[-1]:,.2f}")

    st.caption("Note: This predicts next-period direction using the last lookback window (monthly data).")

# -----------------------------
# Optional: Features table
# -----------------------------
with st.expander("📌 Show Feature Table (Close, SMA3, SMA6, RSI6)"):
    st.dataframe(df[["Date", "Close", "SMA3", "SMA6", "RSI6"]].tail(30), use_container_width=True)