import streamlit as st 
import plotly.graph_objects as go
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# --- SETUP PATHS ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# --- IMPORTS ---
from model.generator import QuantumGenerator
from model.discriminator import Discriminator
from utils.fetch import get_vfv_data, sync_market_clock
from utils.process import get_processed_tensors

# --- CONFIGURATION ---
WEIGHTS_PATH = "vfv_wgan_final.pt"
RISE_THRESHOLD = 0.05
DROP_THRESHOLD = -0.05
CONFIRMATION_MINUTES = 3
ANOMALY_THRESHOLD = -0.5

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Quantum Market Monitor for VFV.TO",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM STYLE (Dark Tech Theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #30333d;
    }
    .metric-label { color: #8b92a6; font-size: 14px; }
    .metric-value { color: #ffffff; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. CACHED MODEL LOADER ---
@st.cache_resource
def load_brain():
    gen = QuantumGenerator()
    disc = Discriminator()
    
    if os.path.exists(WEIGHTS_PATH):
        gen.load_state_dict(torch.load(WEIGHTS_PATH))
        gen.eval()
        disc.eval()
        return gen, disc
    else:
        st.error(f"CRITICAL: Weights file '{WEIGHTS_PATH}' missing.")
        st.stop()

# --- 2. SESSION STATE (Persistence) ---
if 'rising_streak' not in st.session_state:
    st.session_state.rising_streak = 0
if 'dropping_streak' not in st.session_state:
    st.session_state.dropping_streak = 0
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'log_reset_at' not in st.session_state:
    # Used to clear the log every hour (ET)
    st.session_state.log_reset_at = datetime.now(ZoneInfo("America/New_York"))

# --- 3. VISUALIZATION HELPER ---
def plot_quantum_cloud(real_window, futures):
    """
    Plots Reality vs. The Quantum Cloud (Mean + Volatility Cone).
    """
    fig = go.Figure()
    
    # 1. Setup Data
    real_data = real_window.squeeze().numpy()
    x_real = list(range(0, 15))
    x_future = list(range(14, 29)) # Start where reality ends
    
    # 2. Calculate Statistics
    mean_future = np.mean(futures, axis=0)
    std_future = np.std(futures, axis=0)
    
    # Align the future to the last real data point
    offset = real_data[-1] - mean_future[0]
    aligned_mean = mean_future + offset
    
    # Create Upper/Lower bounds for the cloud (2 Standard Deviations)
    upper_bound = aligned_mean + (std_future * 2)
    lower_bound = aligned_mean - (std_future * 2)

    # 3. PLOT: The Volatility Cloud (Shaded Area)
    fig.add_trace(go.Scatter(
        x=x_future + x_future[::-1], # Loop back to close the shape
        y=np.concatenate([upper_bound, lower_bound[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 255, 0.2)', # Transparent Magenta
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Volatility Range'
    ))

    # 4. PLOT: The Real Market (Cyan)
    fig.add_trace(go.Scatter(
        x=x_real, y=real_data, 
        mode='lines+markers', 
        name='Real Market',
        line=dict(color='#00d4ff', width=3)
    ))

    # 5. PLOT: The Quantum Trend (Bright Magenta Dotted)
    fig.add_trace(go.Scatter(
        x=x_future, y=aligned_mean, 
        mode='lines', 
        name='Averaged Quantum Trend (2SD)',
        line=dict(color='#ff00ff', width=3, dash='dot')
    ))

    fig.update_layout(
        title=None, 
        xaxis_title="Time Steps (Minutes)", 
        yaxis_title="Momentum",
        template="plotly_dark", 
        height=480, 
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig
    
# --- 4. MAIN APP LOOP ---
def main():
    # Header
    c1, c2 = st.columns([4, 1])
    with c1: st.title("Quantum Market Monitor for VFV.TO")
    with c2: 
        if st.button("Stop Engine"): st.stop()

    gen, disc = load_brain()

    # Layout Containers (To hold live updates)
    status_area = st.empty()
    metrics_area = st.empty()
    chart_area = st.empty()
    log_area = st.expander("Live Signal Log", expanded=True)
    with log_area:
        log_text_area = st.empty()

    # --- THE LIVE LOOP ---
    while True:
        # A. Sync & Fetch
        with st.spinner("Syncing with Market Clock..."):
            sync_market_clock() 
            get_vfv_data(force_refresh=True)

        # B. Process
        all_tensors = get_processed_tensors()
        if all_tensors is None or all_tensors.shape[0] == 0:
            status_area.warning("Buffer filling... (Wait 1 min)")
            time.sleep(10)
            continue
            
        last_window = all_tensors[-1].unsqueeze(0)

        # C. Inference
        with torch.no_grad():
            critic_score = disc(last_window).item()
            futures = gen(batch_size=200).detach().numpy()
            trend = np.mean(futures)

        # D. Logic Engine (Swing Mode)
        instant_signal = "FLAT"
        if trend > RISE_THRESHOLD: instant_signal = "RISING"
        elif trend < DROP_THRESHOLD: instant_signal = "DROPPING"

        # Update Persistence
        if instant_signal == "RISING":
            st.session_state.rising_streak += 1
            st.session_state.dropping_streak = 0
        elif instant_signal == "DROPPING":
            st.session_state.dropping_streak += 1
            st.session_state.rising_streak = 0
        else:
            st.session_state.rising_streak = 0
            st.session_state.dropping_streak = 0

        # Determine Official Status
        final_status = "HOLD / NEUTRAL"
        box_color = "#2b2d3e" # Default Gray
        
        if st.session_state.rising_streak >= CONFIRMATION_MINUTES:
            final_status = "BUY / UPTREND DETECTED"
            box_color = "#006400" # Dark Green
        elif st.session_state.dropping_streak >= CONFIRMATION_MINUTES:
            final_status = "SELL / DOWNTREND DETECTED"
            box_color = "#8b0000" # Dark Red
            
        # Anomaly Override
        if critic_score < ANOMALY_THRESHOLD:
            final_status = "CRASH WARNING (EXIT)"
            box_color = "#ff0000" # Bright Red

        # --- E. RENDER UI ---
        # Current Eastern Time (handles EST/EDT automatically)
        est_now = datetime.now(ZoneInfo("America/New_York"))
        est_stamp = est_now.strftime("%Y-%m-%d %H:%M:%S %Z")

        # Clear log every hour to prevent unbounded growth
        if (est_now - st.session_state.log_reset_at).total_seconds() >= 3600:
            st.session_state.logs = []
            st.session_state.log_reset_at = est_now
        
        # 1. Status Box
        status_area.markdown(f"""
            <div class="status-box" style="background-color: {box_color};">
                <h1 style="margin:0; color:white;">{final_status}</h1>
                <p style="margin:0; color:#ddd;">Market Health Score: {critic_score:.4f} &nbsp; | &nbsp; Time (ET): {est_stamp}</p>
            </div>
        """, unsafe_allow_html=True)

        # 2. Metrics Grid
        with metrics_area.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Quantum Trend", f"{trend:.4f}")
            c2.metric("Instant Signal", instant_signal)
            c3.metric("Confirm Rise", f"{st.session_state.rising_streak}/{CONFIRMATION_MINUTES}")
            c4.metric("Confirm Drop", f"{st.session_state.dropping_streak}/{CONFIRMATION_MINUTES}")

        # 3. Chart
        fig = plot_quantum_cloud(last_window, futures)
        chart_area.plotly_chart(fig, use_container_width=True)

        # 4. Logs
        ts = est_stamp
        log_entry = f"[{ts}] {instant_signal:<8} | Trend: {trend:+.4f} | Score: {critic_score:.4f}"
        st.session_state.logs.insert(0, log_entry)
        if len(st.session_state.logs) > 8: st.session_state.logs.pop()

        # Replace log contents (don't stack)
        log_text_area.text("\n".join(st.session_state.logs))

        # Loop delay is handled by sync_market_clock(), but we add a tiny safety sleep
        time.sleep(1)

if __name__ == "__main__":
    main()