import pickle
import warnings
from pathlib import Path
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Traffic Congestion Predictor", page_icon="🚦", layout="centered")
ARTIFACTS_DIR = Path("artifacts")

# ── load model artifacts ──────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    def load(name):
        with open(ARTIFACTS_DIR / name, "rb") as f:
            return pickle.load(f)
    return load("model.pkl"), load("encoders.pkl"), load("power_transformer.pkl"), \
           load("skew_cols.pkl"), load("feature_names.pkl")

# ── helpers ───────────────────────────────────────────────────────────────────

def time_block(hour):
    if hour < 6:   return "0-5"
    if hour < 12:  return "6-11"
    if hour < 18:  return "12-17"
    return "18-23"

def congestion_label(speed):
    if speed >= 30: return "🟢 Free Flow",        "#28a745"
    if speed >= 25: return "🟡 Moderate",          "#ffc107"
    if speed >= 20: return "🟠 Heavy Congestion",  "#fd7e14"
    return           "🔴 Severe Congestion",        "#dc3545"

def predict(hour, day_of_week, direction, total_flow, avg_occupancy,
            model, encoders, pt, skew_cols, feature_names):

    row = {
        # user inputs
        "Hour":           hour,
        "Day_of_Week":    day_of_week,
        "Direction":      direction,
        "Total_Flow":     total_flow,
        "Avg_Occupancy":  avg_occupancy,
        # derived
        "Is_Weekend":     "Yes" if day_of_week >= 5 else "No",
        "Time_Block":     time_block(hour),
        # fixed defaults (median-ish values)
        "Day":            1,
        "Month":          1,
        "Lane_Type":      "ML",
        "Station":        717960,
        "Freeway":        5,
        "Station_Length": 0.5,
        "Samples":        60,
        "Percent_Observed": 100.0,
        "Lane1_Flow":     total_flow // 3,
        "Lane1_Avg_Occupancy": avg_occupancy,
        "Lane2_Flow":     total_flow // 3,
        "Lane2_Avg_Occupancy": avg_occupancy,
        "Lane3_Flow":     total_flow // 4,
        "Lane3_Avg_Occupancy": avg_occupancy,
        "Lane1_Samples":  60,
        "Lane2_Samples":  60,
        "Lane3_Samples":  60,
    }

    df = pd.DataFrame([row])

    # encode categoricals
    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col].astype(str))

    # fill any missing features then reorder
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # power transform
    cols_to_transform = [c for c in skew_cols if c in df.columns]
    if cols_to_transform:
        df[cols_to_transform] = pt.transform(df[cols_to_transform])

    return float(model.predict(df)[0])

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🚦 Traffic Congestion Predictor")
st.caption("Predicts average freeway speed using California PeMS D7 sensor data.")
with st.sidebar:
    with st.expander('INFO ℹ️'):
        st.subheader('Developed by :red[RAAKESH A S]')
        st.link_button(':blue[LinkedIn ID]','https://www.linkedin.com/in/raakesh-a-s')
        st.link_button(':red[GitHub]','https://github.com/raakeshcsd1317')

try:
    model, encoders, pt, skew_cols, feature_names = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python train.py --data <file.txt>` first.")
    st.stop()

st.divider()

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("🕐 Hour of Day", 0, 23, 8,
                     help="Hour when the reading is taken (0 = midnight, 8 = 8 AM)")
    day_of_week = st.selectbox("📅 Day of Week",
                               options=list(range(7)),
                               format_func=lambda x: ["Monday","Tuesday","Wednesday",
                                                       "Thursday","Friday","Saturday","Sunday"][x])

with col2:
    direction = st.selectbox("🧭 Direction",
                             options=list(encoders["Direction"].classes_),
                             help="N = Northbound, S = Southbound, E = Eastbound, W = Westbound")
    total_flow = st.number_input("🚗 Total Flow (vehicles / 5 min)", 0, 5000, 300, step=10)

avg_occupancy = st.slider("📈 Avg Occupancy (%)", 0.0, 100.0, 10.0, step=0.5,
                          help="Higher occupancy = more congestion")

st.divider()

if st.button("🚀 Predict", use_container_width=True, type="primary"):
    speed = predict(hour, day_of_week, direction, total_flow, avg_occupancy,
                    model, encoders, pt, skew_cols, feature_names)
    label, color = congestion_label(speed)

    c1, c2 = st.columns(2)
    c1.metric("Predicted Avg Speed", f"{speed:.1f} mph")
    c2.markdown(f"**Congestion Level**<br><span style='color:{color};font-size:1.5rem'>{label}</span>",
                unsafe_allow_html=True)

    st.caption(f"Time Block: {time_block(hour)}  •  "
               f"Weekend: {'Yes' if day_of_week >= 5 else 'No'}")

with st.expander("ℹ️ About"):
    st.markdown("""
**Model:** Random Forest Regressor &nbsp;|&nbsp; **R² = 0.99 · RMSE = 0.72 · MAE = 0.15**

| Input | Why it matters |
|-------|---------------|
| Hour of Day | Peak hours (7–9 AM, 5–7 PM) drive congestion |
| Day of Week | Weekdays vs weekends differ significantly |
| Direction | Traffic is asymmetric by direction |
| Total Flow | More vehicles = higher congestion risk |
| Avg Occupancy | Best single indicator of road saturation |

Other features (lane-level, station metadata) are set to typical defaults internally.
    """)