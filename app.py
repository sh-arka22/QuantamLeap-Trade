# File: app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

API_URL = "http://127.0.0.1:5000"

# === Custom CSS ===
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b, #0f172a);
    color: #e2e8f0;
}
.glass-card {
    background: rgba(30, 41, 59, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# === Sidebar: Collapsible Real-Time Dashboard ===
with st.sidebar.expander("📊 Real-Time Dashboard", expanded=True):
    st.markdown("## Portfolio Stats")
    try:
        stats = requests.get(f"{API_URL}/portfolio").json()
        st.markdown(f"- **Value:** ${stats.get('portfolio_value', 0):,.2f}")
        st.markdown(f"- **P/L:** ${stats.get('total_profit', 0):,.2f}")
        st.markdown(f"- **Win Rate:** {stats.get('win_rate', 0):.1f}%")
    except:
        st.error("Could not fetch portfolio stats.")

    st.markdown("---")
    st.markdown("## Model Status")
    try:
        m = requests.get(f"{API_URL}/model_status").json()
        st.markdown(f"- **Status:** {m.get('status', 'N/A')}")
        st.markdown(f"- **Accuracy:** {m.get('accuracy', 0):.1f}%")
    except:
        st.error("Could not fetch model status.")

    st.markdown("---")
    st.markdown("## Market Trends")
    try:
        symbols = requests.get(f"{API_URL}/csv_files").json()
        for sym in symbols:
            path = os.path.join("backend", "data", f"{sym}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['Date'])
                last = df.sort_values('Date').iloc[-1]
                st.markdown(f"**{sym}**: ${last['Close']:.2f} ({last['Date'].date()})")
    except:
        st.error("Could not load market data.")

# # === Main Navigation ===
# st.title("TradeAI Platform")
# page = st.radio("", ["Trading Dashboard", "Training & Forecast", "Strategy Execution", "Performance Analytics"], horizontal=True)

page = st.radio(
    "Navigate to", 
    ["Trading Dashboard", "Training & Forecast", "Strategy Execution", "Performance Analytics"],
    horizontal=True,
    label_visibility="hidden"
)

# Fetch symbol list (without ".csv")
try:
    symbols = requests.get(f"{API_URL}/csv_files").json()
except:
    symbols = []

# --- Page 1: Trading Dashboard ---
if page == "Trading Dashboard":
    st.header("📈 Trading Dashboard")
    stock = st.selectbox("Select Stock", symbols)
    if stock:
        csv_path = os.path.join("backend", "data", f"{stock}.csv")
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df['MA20'] = df['Close'].rolling(20).mean()
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['BB_Upper'] = df['MA20'] + 2 * df['Close'].rolling(20).std()
            df['BB_Lower'] = df['MA20'] - 2 * df['Close'].rolling(20).std()

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name=stock))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name="MA20"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], name="EMA20"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name="Upper BB", line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name="Lower BB", line=dict(dash='dash')))
            fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading data for {stock}: {e}")

# --- Page 2: Training & Forecast ---
elif page == "Training & Forecast":
    st.header("🛠️ Model Training & Forecast")
    col1, col2 = st.columns(2)
    with col1:
        stock = st.selectbox("Choose Existing CSV", symbols)
    with col2:
        uploaded = st.file_uploader("…or upload CSV", type="csv")
    if uploaded:
        try:
            r = requests.post(
                f"{API_URL}/upload_csv",
                files={"file": (uploaded.name, uploaded, "text/csv")}
            )
            if r.ok:
                st.success(f"Uploaded {uploaded.name}")
                new_sym = os.path.splitext(uploaded.name)[0]
                if new_sym not in symbols:
                    symbols.append(new_sym)
            else:
                st.error("Upload failed.")
        except Exception as e:
            st.error(f"Upload error: {e}")
        stock = os.path.splitext(uploaded.name)[0]

    st.markdown("---")
    time_window = st.number_input("Forecast Horizon (days)", min_value=1, value=30)
    epochs = st.number_input("Epochs", min_value=1, value=20)
    batch_size = st.number_input("Batch Size", min_value=1, value=32)
    if st.button("Train Model"):
        try:
            payload = {"stock": stock, "time_window": time_window, "epochs": epochs, "batch_size": batch_size}
            res = requests.post(f"{API_URL}/train_model", json=payload)
            result = res.json()
            st.success(result.get("message", "Training complete"))
        except Exception as e:
            st.error(f"Training error: {e}")

    if st.button("Forecast OHLCV"):
        try:
            res = requests.post(f"{API_URL}/predict", json={"stock": stock, "days": time_window})
            forecast = res.json()
            df_pred = pd.DataFrame(forecast)
            st.subheader("Forecasted OHLCV")
            st.dataframe(df_pred)
            fig2 = go.Figure(go.Candlestick(
                x=df_pred.index, open=df_pred['Open'], high=df_pred['High'],
                low=df_pred['Low'], close=df_pred['Close']
            ))
            fig2.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=400)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Forecast error: {e}")

# --- Page 3: Strategy Execuation ---
elif page == "Strategy Execution":
    st.header("🤖 Automated Strategy Execution")

    strategy = st.selectbox(
        "Select Strategy",
        ["Evolution Agent", "MA Crossover", "ABCD Strategy", "CNN-LSTM", "Signal Rolling"]
    )
    symbol = st.selectbox("Stock Symbol", symbols)
    days = st.number_input("Use first N forecast days", min_value=1, value=30)
    capital = st.number_input("Initial Capital ($)", min_value=100.0, value=10000.0)

    if st.button("Run Simulation"):
        try:
            payload = {
                "strategy": strategy,
                "symbol": symbol,
                "initial_capital": capital,
                "days": days
            }
            res = requests.post(f"{API_URL}/run_simulation", json=payload)
            res.raise_for_status()
            result = res.json()
            print(result)
            # st.write(result)  # uncomment to display the raw response

            # # --- Historical Simulation ---
            # st.subheader("📈 Historical Simulation")
            # hist = result.get("historical", {})
            # if hist.get("error"):
            #     st.error(hist["error"])
            # else:
            #     # display metrics
            #     st.markdown(f"- **Final Portfolio:** ${hist.get('final_portfolio', 'N/A')}")
            #     st.markdown(f"- **Total Profit:** ${hist.get('total_profit', 'N/A')}")
            #     st.markdown(f"- **Gains:** ${hist.get('gains', 'N/A')}")
            #     st.markdown(f"- **P/L %:** {hist.get('pnl_pct', 'N/A')}%")
            #     st.markdown(f"- **Win Rate:** {hist.get('win_rate', 'N/A')}%")
            #     st.markdown(f"- **Trades Executed:** {hist.get('trades_executed', 0)}")

            #     # portfolio chart
            #     vals = hist.get("portfolio_values", [])
            #     if vals:
            #         df_hist_vals = pd.DataFrame({"Value": vals})
            #         st.line_chart(df_hist_vals)

            #     # trade log
            #     trades = hist.get("trades", [])
            #     if trades:
            #         st.dataframe(pd.DataFrame(trades))

            # st.markdown("---")

            # --- Predicted Simulation ---
            st.subheader(f"🤖 {result.get('strategy')}")
            pred = result.get("predicted", {})
            if result.get("error"):
                st.error(result["error"])
            else:
                # display metrics
                st.markdown(f"- **Final Balance:** ${result.get('final_portfolio')}")
                st.markdown(f"- **Profit:** ${result.get('profit')} ({result.get('gains')}%)")
                st.markdown(f"- **Sharpe Ratio:** {result.get('sharpe_ratio', 'N/A')}")
                st.markdown(f"- **Max Drawdown:** {result.get('max_drawdown_pct')}%")
                st.markdown(f"- **Win Rate:** {result.get('win_rate', 'N/A')}%")
                st.markdown(f"- **Trades Executed:** {result.get('trades_executed', 0)}")

                # portfolio chart
                pvals = pred.get("portfolio_values", [])
                if pvals:
                    df_pred_vals = pd.DataFrame({"Value": pvals})
                    st.line_chart(df_pred_vals)

                # trade log
                ptrades = pred.get("trades", [])
                if ptrades:
                    st.dataframe(pd.DataFrame(ptrades))

        except requests.HTTPError as he:
            st.error(f"API error: {he}")
        except Exception as e:
            st.error(f"Simulation error: {e}")

# --- Page 4: Performance Analytics ---
elif page == "Performance Analytics":
    st.header("📊 Performance Analytics")
    try:
        stats = requests.get(f"{API_URL}/strategy_performance").json()
        for s in stats:
            st.markdown(f"**{s['name']}**: Profit ${s['profit']:.2f}, Win Rate {s['win_rate']:.1f}%")
    except Exception as e:
        st.error(f"Could not fetch performance: {e}")
