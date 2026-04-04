# Israel Alerts Dashboard

Live Streamlit dashboard for Israel civil defense alerts (sirens, pre-alerts, all-clears).

**Live app:** [emergency-alerts-stats.streamlit.app](https://emergency-alerts-stats.streamlit.app)
**Data source:** [github.com/dleshem/israel-alerts-data](https://github.com/dleshem/israel-alerts-data)

---

## Local setup

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

> If you don't have Python 3.10+, install via [Homebrew](https://brew.sh): `brew install python@3.12`

### Linux (Ubuntu / Debian)

```bash
git clone https://github.com/lleviraz/israel-emergency-alerts-stats.git
cd israel-emergency-alerts-stats

sudo apt update && sudo apt install python3 python3-venv python3-pip -y

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at **http://localhost:8501**

---

## Local development mode

The deployed app enforces a 1-hour cooldown on data refresh and model training.
To disable this locally, create a `.env` file in the project root (it is gitignored):

```
LOCAL_MODE=true
DEFAULT_AREA=Your City
```

Load the variables before running:

```bash
# macOS / Linux
export LOCAL_MODE=true
export DEFAULT_AREA="Kfar Netter"
streamlit run dashboard.py

# Windows PowerShell
$env:LOCAL_MODE="true"
$env:DEFAULT_AREA="Kfar Netter"
streamlit run dashboard.py
```

---

## First use

1. Click **⬇ Load / Refresh Data** in the sidebar — downloads the ~50 MB dataset.
2. Select an area and a time range.
3. **Area Analysis tab** — timing stats, convergence rate, risk windows, ML prediction.
4. To use **🚨 Simulate a pre-alert event now**, first click **🧠 Train models** in the sidebar.
5. **Overview tab** — global timeline, top locations, siren heatmap (all areas).
6. **Compare Areas tab** — side-by-side statistics and convergence chart for 2–5 areas.

---

## Running tests

```bash
pip install pytest
pytest tests/
```
