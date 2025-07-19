# File: README.md
# Trading-App-V2

A neural trading platform for algorithmic trading that includes:

- **Real-time Dashboard:** Collapsible sidebar displaying market trends, portfolio statistics (value, P&L, win rate).
- **Model Training & Forecasting:** Upload/select stock OHLCV CSV data, train forecasting model (e.g., CNN-LSTM), and predict 30-day future prices.
- **Automated Strategy Execution:** Choose trading strategy (MA, RSI, Evolutionary, CNN-LSTM, etc.), run simulation on forecast for N days, and view live P&L updates.
- **Performance Analytics:** Compare backtest results and strategy performance metrics.
- **Technical Indicators:** Charts with SMA, EMA, Bollinger Bands overlays for analysis.
- **Backend:** Modular Flask API with endpoints for data upload, training, prediction, and simulation.
- **Frontend:** Streamlit-based UI with Plotly charts and real-time updates.

## Installation

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd Trading-App-V2
