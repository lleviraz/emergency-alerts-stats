# Israel Alerts Dashboard

Live Streamlit dashboard for analyzing Israel civil defense alerts (sirens, pre-alerts, all-clears, all-clear releases).

**Live app:** [emergency-alerts-stats.streamlit.app](https://emergency-alerts-stats.streamlit.app)
**Data source:** [github.com/dleshem/israel-alerts-data](https://github.com/dleshem/israel-alerts-data) (Apache 2.0 license)

---

## ⚠️ Disclaimer

**This dashboard is provided AS-IS for informational and analytical purposes only.**

- **No Warranty:** The authors provide no warranty of any kind regarding the accuracy, completeness, or reliability of the data, analysis, or predictions.
- **Data Source:** Alert data is sourced from [dleshem/israel-alerts-data](https://github.com/dleshem/israel-alerts-data), which aggregates publicly available civil defense alerts. The accuracy and timeliness of this data are not guaranteed.
- **Not Official:** This is an independent analysis tool and is **not affiliated with or endorsed by any government or official civil defense agency**.
- **No Liability:** Users of this dashboard assume all risk. The authors are not liable for any direct, indirect, or consequential damages arising from its use.
- **Predictions:** Machine learning predictions (convergence rate, timing distributions) are statistical estimates based on historical data and **should not be relied upon as ground truth** for decision-making in emergency situations.

**For official civil defense guidance, always refer to your country's official authorities.**

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
export DEFAULT_AREA="Tel Aviv"
streamlit run dashboard.py

# Windows PowerShell
$env:LOCAL_MODE="true"
$env:DEFAULT_AREA="Tel Aviv"
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
