# Israel Alerts Dashboard

Live Streamlit dashboard for Israel civil defense alerts (sirens, pre-alerts, all-clears).

Data source: [github.com/dleshem/israel-alerts-data](https://github.com/dleshem/israel-alerts-data)

## Requirements

- Python 3.10+

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run dashboard.py
```

Opens at **http://localhost:8501**

## First use

1. Click **⬇ Load / Refresh Data** in the sidebar — downloads the ~50 MB dataset.
2. Select an area (defaults to *Kfar Netter*) and a time range.
3. **Area Analysis tab** — timing stats, convergence rate, risk windows, ML prediction.
4. To use **🚨 Pre-alert now...**, first click **🧠 Train model** in the sidebar.
5. **Overview tab** — global timeline, top locations, heatmap (unfiltered by area).
