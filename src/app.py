
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, IsolationForest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "splits"

st.set_page_config(page_title="Cerberus", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background:#0b1120;color:#e5eaf3}
.block-container {max-width:1200px;padding-top:1.5rem}
[data-testid="stMetric"] {
    background:#131e30;border:1px solid #29374e;
    border-radius:10px;padding:16px
}
div.stButton > button {
    width:100%;height:44px;border-radius:8px;font-weight:600
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    X = pd.read_csv(DATA / "X_train.csv")
    y = pd.read_csv(DATA / "y_train.csv").iloc[:, 0]
    X_test = pd.read_csv(DATA / "X_test.csv")
    y_test = pd.read_csv(DATA / "y_test.csv").iloc[:, 0]

    X = X.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    medians = X.median().fillna(0)
    X = X.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        random_state=42, n_jobs=-1
    )
    rf.fit(X, y)

    columns = [
        c for c in X.columns
        if "port" not in c.lower()
        and "win_bytes" not in c.lower()
    ]

    benign = X[y.astype(str).str.lower() == "benign"]
    benign = benign[columns]
    benign = np.sign(benign) * np.log1p(np.abs(benign))

    iso = IsolationForest(
        n_estimators=100, max_samples=1024,
        random_state=42, n_jobs=-1
    )
    iso.fit(benign)

    scores = -iso.decision_function(benign)
    threshold = float(np.quantile(scores, 0.90))

    return rf, iso, columns, medians, threshold, X_test, y_test


if "mode" not in st.session_state:
    st.session_state.mode = "Normal"
if "events" not in st.session_state:
    st.session_state.events = []
if "cursor" not in st.session_state:
    st.session_state.cursor = 0
if "total" not in st.session_state:
    st.session_state.total = 0
if "flags" not in st.session_state:
    st.session_state.flags = 0
if "attacks" not in st.session_state:
    st.session_state.attacks = 0

st.title(" Cerberus ")
st.caption("DDoS Detection · Offline Traffic Replay")
st.divider()

try:
    rf, iso, columns, medians, threshold, X_test, y_test = load_models()
except Exception as e:
    st.error(f"Could not load models or data: {e}")
    st.stop()

benign_ids = np.flatnonzero(
    y_test.astype(str).str.lower().eq("benign").to_numpy()
)
ddos_ids = np.flatnonzero(
    y_test.astype(str).str.lower().eq("ddos").to_numpy()
)

if len(benign_ids) == 0 or len(ddos_ids) == 0:
    st.error("Test data must contain benign and DDoS records.")
    st.stop()

c1, c2, c3 = st.columns(3)

if c1.button("▶ Simulate DDoS", type="primary"):
    st.session_state.mode = "DDoS"
    st.session_state.events = []
    st.session_state.cursor = 0
    st.session_state.total = 0
    st.session_state.flags = 0
    st.session_state.attacks = 0

if c2.button("■ Stop"):
    st.session_state.mode = "Stopped"

if c3.button("↻ Reset"):
    st.session_state.mode = "Normal"
    st.session_state.events = []
    st.session_state.cursor = 0
    st.session_state.total = 0
    st.session_state.flags = 0
    st.session_state.attacks = 0

if st.session_state.mode == "Normal":
    st.success("NORMAL BASELINE · Replaying benign traffic")
elif st.session_state.mode == "DDoS":
    st.error("DDoS SIMULATION · Replaying DDoS dataset records")
else:
    st.warning("SIMULATION STOPPED")

st.caption("Offline replay only. No live packets are captured or transmitted.")


@st.fragment(run_every="1s")
def live_monitor():
    mode = st.session_state.mode

    if mode in ("Normal", "DDoS"):
        ids = benign_ids if mode == "Normal" else ddos_ids
        start = st.session_state.cursor
        chosen = ids[start:start + 5]

        if len(chosen) == 0:
            start = 0
            chosen = ids[:5]

        batch = X_test.iloc[chosen].copy()
        labels = y_test.iloc[chosen].astype(str).tolist()

        predictions = rf.predict(batch)

        anomaly_data = batch[columns]
        anomaly_data = np.sign(anomaly_data) * np.log1p(
            np.abs(anomaly_data)
        )
        scores = -iso.decision_function(anomaly_data)
        flags = scores > threshold

        new_events = []
        for i in range(len(batch)):
            new_events.append({
                "Time": time.strftime("%H:%M:%S"),
                "Traffic": mode,
                "Dataset Label": labels[i],
                "RF Prediction": str(predictions[i]),
                "Anomaly Score": round(float(scores[i]), 4),
                "Anomaly Flag": bool(flags[i])
            })

        st.session_state.events = (
            new_events + st.session_state.events
        )[:200]

        st.session_state.cursor = start + len(batch)
        st.session_state.total += len(batch)
        st.session_state.flags += int(flags.sum())
        st.session_state.attacks += sum(
            str(p).lower() != "benign" for p in predictions
        )

    a, b, c = st.columns(3)
    a.metric("Records Processed", st.session_state.total)
    b.metric("Anomalies Flagged", st.session_state.flags)
    c.metric("RF Attack Predictions", st.session_state.attacks)

    events = pd.DataFrame(st.session_state.events)

    st.subheader("Live Traffic Analytics")

    if not events.empty:
        graph_data = events.iloc[::-1].reset_index(drop=True)
        graph_data["Record"] = range(1, len(graph_data) + 1)

        left, right = st.columns(2)

        with left:
            st.caption("Anomaly Score vs Threshold")
            scores_chart = graph_data[
                ["Record", "Anomaly Score"]
            ].set_index("Record")
            scores_chart["Threshold"] = threshold
            st.line_chart(scores_chart, height=260)

        with right:
            st.caption("Detection Activity")
            activity = pd.DataFrame({
                "Count": [
                    int(
                        events["RF Prediction"]
                        .astype(str).str.lower().ne("benign").sum()
                    ),
                    int(events["Anomaly Flag"].sum())
                ]
            }, index=["RF attack predictions", "Anomaly flags"])
            st.bar_chart(activity, height=260)
    else:
        st.info("Graphs will appear when traffic records arrive.")

    st.subheader("Incoming Traffic")

    if not events.empty:
        st.dataframe(events.head(10), use_container_width=True, hide_index=True)
    else:
        st.caption("Waiting for traffic...")

    with st.expander("Detection Event Log"):
        if not events.empty:
            st.dataframe(events, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Event Log",
                events.to_csv(index=False).encode("utf-8"),
                "netguard_event_log.csv",
                "text/csv"
            )

    st.caption(f"Isolation Forest anomaly threshold: {threshold:.4f}")


live_monitor()
