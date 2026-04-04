# Israel Alerts Dashboard

Live Streamlit dashboard for Israel civil defense alerts (sirens, pre-alerts, all-clears).

Data source: [github.com/dleshem/israel-alerts-data](https://github.com/dleshem/israel-alerts-data)

---

## Requirements

- Python 3.10+
- Git (to clone the repo)

---

## Setup

### Windows

```powershell
git clone https://github.com/lleviraz/israel-emergency-alerts-stats.git
cd israel-emergency-alerts-stats

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run dashboard.py
```

### macOS

```bash
git clone https://github.com/lleviraz/israel-emergency-alerts-stats.git
cd israel-emergency-alerts-stats

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard.py
```

> If you don't have Python 3.10+, install it via [Homebrew](https://brew.sh):
> ```bash
> brew install python@3.12
> ```

### Linux (Ubuntu / Debian)

```bash
git clone https://github.com/lleviraz/israel-emergency-alerts-stats.git
cd israel-emergency-alerts-stats

# Install Python if needed
sudo apt update && sudo apt install python3 python3-venv python3-pip -y

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at **http://localhost:8501**

---

## Local development mode

By default the app enforces a 1-hour cooldown on data refresh and model training (to be fair to all users when deployed). To disable this locally, create a `.env` file in the project root:

```
LOCAL_MODE=true
```

Then load it before running:

```bash
# macOS / Linux
export LOCAL_MODE=true
streamlit run dashboard.py

# Windows PowerShell
$env:LOCAL_MODE="true"
streamlit run dashboard.py
```

---

## First use

1. Click **⬇ Load / Refresh Data** in the sidebar — downloads the ~50 MB dataset.
2. Select an area (defaults to *Kfar Netter*) and a time range.
3. **Area Analysis tab** — timing stats, convergence rate, risk windows, ML prediction.
4. To use **🚨 Simulate a pre-alert event now**, first click **🧠 Train models** in the sidebar.
5. **Overview tab** — global timeline, top locations, siren heatmap (all areas).
6. **Compare Areas tab** — side-by-side statistics and convergence chart for 2–5 areas.

---

## Network setup (share on Wi-Fi)

The app binds to `0.0.0.0`, so anyone on the same Wi-Fi can reach it once the steps below are done.

### Find your local IP

| OS | Command |
|---|---|
| Windows | `ipconfig` → look for **IPv4 Address** under your Wi-Fi adapter |
| macOS | `ipconfig getifaddr en0` (Wi-Fi) or `ifconfig \| grep "inet "` |
| Linux | `ip a` or `hostname -I` |

Share the address with others on the same network, e.g. `http://192.168.1.233:8501`.

Your router assigns this via DHCP so it may change after a reboot. For a permanent address, log into your router (`http://192.168.1.1`) and assign a static/reserved IP for your machine's MAC address.

---

### Windows — network profile

When Windows first connects to a network it asks whether it's *Public* or *Private*. **Public** blocks all inbound LAN traffic regardless of firewall rules — this is the most common reason phones can't reach the app.

**Settings → Network & Internet → Wi-Fi → your network → Properties → set to Private**

If it's still blocked, add a firewall rule once in PowerShell (Administrator):

```powershell
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501
```

To remove it later:

```powershell
netsh advfirewall firewall delete rule name="Streamlit"
```

---

### macOS — firewall

macOS firewall is usually off by default. If it's on:

**System Settings → Network → Firewall → Options** — add Python or allow incoming connections for port 8501.

Or temporarily from the terminal:

```bash
# Allow port 8501 (requires admin)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add $(which python3)
```

---

### Linux — firewall

**UFW (Ubuntu/Debian):**

```bash
sudo ufw allow 8501/tcp
```

**firewalld (Fedora/RHEL/CentOS):**

```bash
sudo firewall-cmd --add-port=8501/tcp --permanent
sudo firewall-cmd --reload
```

**iptables (any distro):**

```bash
sudo iptables -A INPUT -p tcp --dport 8501 -j ACCEPT
```

---

> **Privacy note:** your home router's NAT means this is local-network only. Nobody on the internet can reach port 8501 on your machine unless you explicitly configure port forwarding — which you should not do.
