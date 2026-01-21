---
title: YFinanceQCBM
emoji: "⚛️"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
short_description: Real-time quantum risk monitor for VFV / S&P 500.
---

# VFV.TO Quantum Market Monitor
A hybrid quantum-classical financial surveillance engine designed to forecast short-term volatility in the S&P 500 (VFV.TO). 
Unlike traditional technical indicators that rely on past averages (SMA, RSI), this project uses a Quantum Generative Adversarial Network (QGAN) to simulate 200 possible future price trajectories in real-time, visualizing the market's "probability cloud" to detect trends and anomalies.

## Project Features
**Model Arch:** Features **Wassserstein GAN** with **Gradient Penalty**.

The quantum generator is 10-Qubit Variational Quantum Circuit that maps random noise to market-like volatility patterns using Angle Embedding and Circular Entanglement, while the classical discriminator neural network trained to distinguish between real market microstructure and quantum-generated simulations.

**User Interface**
- Features a dashboard (Streamlit + Plotly interface) that renders the "Quantum Cloud" (Mean Prediction +/- 2SDs) and triggers Buy/Sell signals based on confirmed regime changes.
- Direct integration with `yfinance` allows for real-time 1-minute candle processing, with Z-Score normalization dynamically adjusted to the current 15-minute window.

**Misc.**
- **Quantum Anomaly Detection:** The discriminator network assigns a "Realism Score" to live market data; scores below the trained threshold (-0.5) trigger an immediate Crash Warning.
- **Swing Trading Logic**: Implements a filtered signal system that requires 3 consecutive minutes of trend in the same direction to differentiate between true market reversals and random fluctuations.

**Disclaimer: AI suggestions may not always be accuracte. Use at own risk.**