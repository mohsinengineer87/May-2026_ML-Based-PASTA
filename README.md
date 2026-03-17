# PASTA-ML: A Scalable Machine Learning-Integrated Threat Modeling Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Research Overview

**PASTA-ML** extends the traditional [PASTA (Process for Attack Simulation and Threat Analysis)](https://threat-modeling.com/pasta-threat-modeling/) framework with machine learning-based risk estimation and rigorous quantitative scalability evaluation — designed for **large-scale cyber-physical and distributed systems**.

### Research Problem
Existing PASTA implementations lack empirical evaluation of how data generation, vulnerability feature extraction, model training, and inference scale with growing numbers of assets and threat vectors. This framework addresses that gap.

### 6-Step Pipeline

| Phase | Step | Description |
|-------|------|-------------|
| Phase 1 | Step 1 | Modified PASTA Framework Design |
| Phase 1 | Step 2 | System Modeling & Threat Environment Simulation |
| Phase 2 | Step 3 | Synthetic Threat Scenario Generation |
| Phase 2 | Step 4 | Feature Engineering & Complexity Characterization |
| Phase 3 | Step 5 | Machine Learning-Based Risk Estimation |
| Phase 3 | Step 6 | Scalability & Performance Evaluation |

---

## 🚀 Quick Start

### Option 1 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/pasta-ml-framework.git
cd pasta-ml-framework

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run pasta_ml_app.py
```

The app will open automatically at `http://localhost:8501`

### Option 2 — Streamlit Cloud (no install needed)

Click the badge at the top of this README, or visit:  
`https://your-app-name.streamlit.app`

---

## 📁 Repository Structure

```
pasta-ml-framework/
│
├── pasta_ml_app.py          ← Main Streamlit application (all 6 steps)
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
├── LICENSE                  ← MIT License
│
└── .streamlit/
    └── config.toml          ← Streamlit theme & server configuration
```

---

## 🔬 Framework Details

### Step 1 — Modified PASTA Framework Design
- Restructures the 7-stage PASTA methodology for automated, scalable analysis
- Maps each stage to quantitative model variables and ML pipeline components
- Theoretical complexity analysis: O(1) → O(N) → O(N²) → NP-hard across stages

### Step 2 — System Environment Simulation
- Simulates **7 asset types**: Cloud VM, IoT Device, Database Server, Network Device, Enterprise App, SCADA/ICS, Endpoint
- **NVD-calibrated** vulnerability counts via Poisson distribution per asset type
- **6 STIX 2.1-aligned threat actor profiles**: APT Group, Nation-State, Cybercriminal, Insider Threat, Hacktivist, Script Kiddie
- CIA impact ratings, patch compliance, control coverage, and network exposure per asset

### Step 3 — Synthetic Threat Scenario Generation
- Generates up to **10,000 attack scenarios** combining assets, vulnerabilities, and threat vectors
- **NetworkX directed attack graph** with Dijkstra-based shortest-path analysis
- **12 MITRE ATT&CK-aligned attack vectors** (Phishing, SQLi, RCE, Lateral Movement, etc.)
- CVSS scores follow **NVD 2024 severity distribution**: Critical 14%, High 34%, Medium 50%, Low 2%

### Step 4 — Feature Engineering (10 Features)

| Feature | Description |
|---------|-------------|
| `asset_criticality` | Weighted CIA impact × exposure factor |
| `vuln_count_norm` | Log-normalised vulnerability count |
| `cvss_weighted_avg` | Severity-weighted average CVSS score |
| `exploitability_score` | CVSS exploitability sub-score composite |
| `attack_path_length_inv` | 1 / shortest path length (shorter = riskier) |
| `threat_likelihood` | Capability × motivation × exposure product |
| `exposure_level` | Network zone ordinal (internet=1.0 → air-gap=0.1) |
| `patch_compliance_inv` | 1 − patch compliance rate |
| `attacker_capability` | Normalised threat actor capability score |
| `control_effectiveness_inv` | 1 − security control coverage |

**Target variable**: Composite Risk Score (0–10) using known weighted formula + calibrated noise → enables ground-truth ML validation.

### Step 5 — ML Risk Estimation
- **Random Forest Regressor** with configurable n_estimators, max_depth
- **Gradient Boosting Regressor** with configurable learning_rate, n_estimators
- Evaluation: R², MAE, RMSE, MAPE, k-fold cross-validation
- **SHAP TreeExplainer** beeswarm plots for feature explainability
- **Permutation importance** with standard deviation error bars

### Step 6 — Scalability Benchmarks
- Profiles **4 pipeline stages** independently: data generation, scenario generation, feature engineering, ML training+inference
- Measures: wall-clock time (`time.perf_counter`), peak memory (`tracemalloc`)
- Computes throughput (scenarios/second)
- Fits **O(N^k) log-log slope** per stage to classify complexity empirically

---

## 📊 Expected Results

| Metric | Expected Range |
|--------|---------------|
| R² (Random Forest) | 0.85 – 0.95 |
| R² (Gradient Boosting) | 0.85 – 0.95 |
| MAE | < 0.5 on 0–10 scale |
| Pipeline Complexity | O(N^1.0) – O(N^1.3) |
| Throughput | 500 – 5,000 scenarios/s |

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web application |
| `plotly` | Interactive charts (all visualisations) |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scikit-learn` | Random Forest, Gradient Boosting, metrics |
| `networkx` | Attack graph construction + path analysis |
| `shap` | ML model explainability (SHAP values) |

---

## 📚 References

1. UcedaVélez, T. & Morana, M.M. (2015). *Risk Centric Threat Modeling*. Wiley.
2. MITRE ATT&CK Framework — https://attack.mitre.org
3. NIST NVD / CVSS v3.1 — https://nvd.nist.gov/vuln-metrics/cvss
4. Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
5. Hagberg, A. et al. (2008). *Exploring Network Structure, Dynamics, and Function using NetworkX*.
6. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
7. Xiong, W. & Lagerström, R. (2019). Threat modeling — A systematic literature review. *Computers & Security*, 84, 53–69.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Research project for quantitative scalability evaluation of the PASTA threat modeling framework.  
Built with Streamlit + scikit-learn + NetworkX + SHAP.
