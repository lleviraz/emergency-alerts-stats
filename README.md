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

## Local network access (same Wi-Fi)

The app binds to `0.0.0.0` so anyone on your Wi-Fi can open it.

1. Find your machine's local IP:
   ```
   ipconfig
   ```
   Look for **IPv4 Address** under your Wi-Fi adapter (e.g. `192.168.1.233`).

2. Allow inbound connections on port 8501 through Windows Firewall — run **once** in PowerShell as Administrator:
   ```powershell
   netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501
   ```

3. Share the URL with anyone on the same Wi-Fi:
   ```
   http://192.168.1.233:8501
   ```
   Replace `192.168.1.233` with your actual IP.

> **Note:** this is local-network only. Your home router's NAT blocks all access from the internet — no extra configuration needed to keep it private.

To remove the firewall rule later:
```powershell
netsh advfirewall firewall delete rule name="Streamlit"
```

## First use

1. Click **⬇ Load / Refresh Data** in the sidebar — downloads the ~50 MB dataset.
2. Select an area (defaults to *Kfar Netter*) and a time range.
3. **Area Analysis tab** — timing stats, convergence rate, risk windows, ML prediction.
4. To use **🚨 Pre-alert now...**, first click **🧠 Train model** in the sidebar.
5. **Overview tab** — global timeline, top locations, heatmap (unfiltered by area).
