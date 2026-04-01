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

## Network setup (share on Wi-Fi)

The app binds to `0.0.0.0`, so anyone on the same Wi-Fi can reach it. Three things need to be in order:

### 1. Windows network profile — most important

When Windows first connects to a network it asks whether it's *Public* or *Private*. **Public** blocks all inbound LAN traffic regardless of firewall rules.

To check / change:
- **Settings → Network & Internet → Wi-Fi → your network name → Properties**
- Set **Network profile type** to **Private**

This is the most common reason phones and other devices can't reach the app.

### 2. Find your local IP

```
ipconfig
```

Look for **IPv4 Address** under your Wi-Fi adapter — e.g. `192.168.1.233`. Share this address with others on the same network:

```
http://192.168.1.233:8501
```

Your router assigns this via DHCP so it may change after a reboot. To make it permanent, log into your router (`http://192.168.1.1`) and assign a static/reserved IP for your machine's MAC address.

### 3. Windows Firewall rule (if still blocked)

If the network profile is set to Private and it's still unreachable, add a firewall rule. Run **once** in PowerShell as Administrator:

```powershell
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501
```

To remove it later:

```powershell
netsh advfirewall firewall delete rule name="Streamlit"
```

> **Privacy note:** your home router's NAT means this is local-network only. Nobody on the internet can reach port 8501 on your machine unless you explicitly configure port forwarding — which you should not do.

## First use

1. Click **⬇ Load / Refresh Data** in the sidebar — downloads the ~50 MB dataset.
2. Select an area (defaults to *Kfar Netter*) and a time range.
3. **Area Analysis tab** — timing stats, convergence rate, risk windows, ML prediction.
4. To use **🚨 Pre-alert now...**, first click **🧠 Train model** in the sidebar.
5. **Overview tab** — global timeline, top locations, heatmap (unfiltered by area).
