"""
PASTA-ML: A Scalable Machine Learning-Integrated Threat Modeling Framework
for Large-Scale Cyber-Physical Systems

6-Step Research Pipeline:
  Phase 1 | Step 1: Modified PASTA Framework Design
           | Step 2: System Modeling & Threat Environment Simulation
  Phase 2 | Step 3: Synthetic Threat Scenario Generation
           | Step 4: Feature Engineering & Complexity Characterization
  Phase 3 | Step 5: Machine Learning-Based Risk Estimation
           | Step 6: Scalability & Performance Evaluation

Run:  streamlit run pasta_ml_app.py
Deps: pip install streamlit plotly pandas numpy scikit-learn networkx shap
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import io, json, time, tracemalloc, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import shap

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ← must be FIRST Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PASTA-ML Research Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.phase-badge{display:inline-block;padding:4px 12px;border-radius:14px;
  font-size:0.78rem;font-weight:700;letter-spacing:0.04em;margin-right:6px;}
.step-card{background:#f0f6ff;border-left:5px solid #1a73e8;
  border-radius:8px;padding:14px 18px;margin:8px 0;}
.formula-box{background:#0f1923;border-left:4px solid #00c4ff;border-radius:6px;
  padding:12px 16px;font-family:monospace;color:#d4f1ff;margin:6px 0;font-size:0.95rem;}
.callout-info{background:#e8f4fb;border-left:4px solid #2196F3;
  padding:10px 14px;border-radius:4px;font-size:0.9rem;margin:6px 0;}
.callout-warn{background:#fff8e1;border-left:4px solid #FFC107;
  padding:10px 14px;border-radius:4px;font-size:0.9rem;margin:6px 0;}
.callout-good{background:#e8f5e9;border-left:4px solid #4CAF50;
  padding:10px 14px;border-radius:4px;font-size:0.9rem;margin:6px 0;}
.metric-pill{display:inline-block;background:#1a73e8;color:white;
  padding:3px 10px;border-radius:10px;font-size:0.8rem;margin:2px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — Asset Types, Threat Actors, Attack Vectors
# ─────────────────────────────────────────────────────────────────────────────
ASSET_TYPES = {
    "Cloud VM":        {"base_vuln_lambda": 20, "base_criticality": 0.65, "exposure": 0.85,
                        "color": "#4285F4", "icon": "☁️"},
    "IoT Device":      {"base_vuln_lambda": 15, "base_criticality": 0.55, "exposure": 0.75,
                        "color": "#EA4335", "icon": "📡"},
    "Database Server": {"base_vuln_lambda": 18, "base_criticality": 0.90, "exposure": 0.50,
                        "color": "#34A853", "icon": "🗄️"},
    "Network Device":  {"base_vuln_lambda":  8, "base_criticality": 0.70, "exposure": 0.70,
                        "color": "#FBBC04", "icon": "🔌"},
    "Enterprise App":  {"base_vuln_lambda": 12, "base_criticality": 0.75, "exposure": 0.60,
                        "color": "#9C27B0", "icon": "🖥️"},
    "SCADA/ICS":       {"base_vuln_lambda": 10, "base_criticality": 0.95, "exposure": 0.40,
                        "color": "#FF5722", "icon": "⚙️"},
    "Endpoint":        {"base_vuln_lambda": 12, "base_criticality": 0.45, "exposure": 0.65,
                        "color": "#00BCD4", "icon": "💻"},
}

THREAT_ACTORS = {
    "APT Group":        {"capability": (7,10), "persistence": (8,10), "motivation": "espionage",
                         "color": "#c0392b", "icon": "🎯"},
    "Nation-State":     {"capability": (8,10), "persistence": (9,10), "motivation": "sabotage",
                         "color": "#8e44ad", "icon": "🏛️"},
    "Cybercriminal":    {"capability": (4,7),  "persistence": (3,6),  "motivation": "financial",
                         "color": "#e67e22", "icon": "💰"},
    "Insider Threat":   {"capability": (3,7),  "persistence": (5,8),  "motivation": "revenge",
                         "color": "#e74c3c", "icon": "👤"},
    "Hacktivist":       {"capability": (3,6),  "persistence": (2,4),  "motivation": "ideology",
                         "color": "#2980b9", "icon": "🌐"},
    "Script Kiddie":    {"capability": (1,3),  "persistence": (1,2),  "motivation": "notoriety",
                         "color": "#7f8c8d", "icon": "💣"},
}

ATTACK_VECTORS = {
    "Phishing":              {"difficulty": 0.25, "tactic": "Initial Access",   "cvss_base": 7.5},
    "SQLi":                  {"difficulty": 0.35, "tactic": "Exploitation",     "cvss_base": 8.2},
    "RCE via Unpatched CVE": {"difficulty": 0.45, "tactic": "Exploitation",     "cvss_base": 9.3},
    "Privilege Escalation":  {"difficulty": 0.50, "tactic": "Privilege Esc.",   "cvss_base": 7.8},
    "Lateral Movement":      {"difficulty": 0.40, "tactic": "Lateral Movement", "cvss_base": 7.0},
    "Credential Stuffing":   {"difficulty": 0.20, "tactic": "Initial Access",   "cvss_base": 6.5},
    "Supply Chain":          {"difficulty": 0.70, "tactic": "Initial Access",   "cvss_base": 9.8},
    "DDoS":                  {"difficulty": 0.15, "tactic": "Impact",           "cvss_base": 5.0},
    "Data Exfiltration":     {"difficulty": 0.55, "tactic": "Exfiltration",     "cvss_base": 8.5},
    "Firmware Implant":      {"difficulty": 0.80, "tactic": "Persistence",      "cvss_base": 9.1},
    "Pass-the-Hash":         {"difficulty": 0.35, "tactic": "Lateral Movement", "cvss_base": 7.2},
    "Zero-Day Exploit":      {"difficulty": 0.90, "tactic": "Exploitation",     "cvss_base": 9.9},
}

PASTA_STAGES = {
    1: {"name":"Define Objectives",        "icon":"🎯", "color":"#1a5276"},
    2: {"name":"Technical Scope",          "icon":"🗺️", "color":"#1f618d"},
    3: {"name":"Decompose Application",    "icon":"🔩", "color":"#2874a6"},
    4: {"name":"Threat Analysis",          "icon":"⚔️", "color":"#17a589"},
    5: {"name":"Vulnerability Analysis",   "icon":"🔍", "color":"#d68910"},
    6: {"name":"Attack Modeling",          "icon":"🕸️", "color":"#ba4a00"},
    7: {"name":"Risk & Impact Analysis",   "icon":"📊", "color":"#7d3c98"},
}

FEATURE_NAMES = [
    "asset_criticality",
    "vuln_count_norm",
    "cvss_weighted_avg",
    "exploitability_score",
    "attack_path_length_inv",
    "threat_likelihood",
    "exposure_level",
    "patch_compliance_inv",
    "attacker_capability",
    "control_effectiveness_inv",
]

FEATURE_DESCRIPTIONS = {
    "asset_criticality":        "Weighted CIA impact × exposure factor (Stage 2)",
    "vuln_count_norm":          "Log-normalised vulnerability count (Stage 5)",
    "cvss_weighted_avg":        "Severity-weighted avg CVSS score (Stage 5)",
    "exploitability_score":     "CVSS exploitability sub-score composite (Stage 5)",
    "attack_path_length_inv":   "1 / shortest path length — shorter = riskier (Stage 6)",
    "threat_likelihood":        "Capability × motivation × exposure product (Stage 4)",
    "exposure_level":           "Network zone ordinal (internet=1 → air-gap=0.1) (Stage 2)",
    "patch_compliance_inv":     "1 − patch compliance rate (Stage 5)",
    "attacker_capability":      "Normalised threat actor capability score (Stage 4)",
    "control_effectiveness_inv":"1 − security control coverage (Stage 7)",
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "env":            None,   # simulated environment (Step 2)
    "scenarios":      None,   # threat scenarios DataFrame (Step 3)
    "features":       None,   # engineered feature DataFrame (Step 4)
    "ml_results":     None,   # trained models + metrics (Step 5)
    "bench_results":  None,   # scalability benchmark DataFrame (Step 6)
    "attack_graph":   None,   # NetworkX DiGraph (Step 3)
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def fig_bytes(fig):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    return buf

def safe_log10(x):
    x = np.asarray(x, dtype=np.float64)
    pos = x[x > 0]
    eps = np.min(pos) * 1e-9 if pos.size > 0 else 1e-12
    return np.log10(np.clip(x, eps, None))

def complexity_class(slope):
    if slope < 1.1:   return "🟢 O(N) — Linear",        "#27ae60"
    if slope < 1.5:   return "🟡 O(N^k) — Near-Linear", "#f39c12"
    if slope < 2.0:   return "🟠 O(N^k) — Super-Linear","#e67e22"
    return              "🔴 O(N²+) — Quadratic+",        "#c0392b"

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 ENGINE — System Environment Simulator
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def simulate_environment(n_assets, seed, asset_mix, threat_actor_types):
    """Build a simulated cyber-physical system environment."""
    rng = np.random.default_rng(seed)
    assets = []
    total_weight = sum(asset_mix.values())
    for asset_type, weight in asset_mix.items():
        n = max(1, int(round(n_assets * weight / total_weight)))
        cfg = ASSET_TYPES[asset_type]
        for i in range(n):
            crit  = float(np.clip(rng.normal(cfg["base_criticality"], 0.12), 0.1, 1.0))
            exp   = float(np.clip(rng.normal(cfg["exposure"],         0.15), 0.1, 1.0))
            patch = float(np.clip(rng.beta(2, 3),                           0.0, 1.0))
            ctrl  = float(np.clip(rng.beta(3, 2),                           0.0, 1.0))
            c_imp = float(np.clip(rng.normal(crit * 0.9, 0.1), 0, 1))
            i_imp = float(np.clip(rng.normal(crit * 0.8, 0.1), 0, 1))
            a_imp = float(np.clip(rng.normal(crit * 0.7, 0.1), 0, 1))
            acs   = (0.4*c_imp + 0.3*i_imp + 0.3*a_imp) * exp
            n_vulns = max(0, int(rng.poisson(cfg["base_vuln_lambda"] * (0.5 + crit))))
            assets.append({
                "asset_id":            f"{asset_type[:3].upper()}-{i:04d}",
                "asset_type":          asset_type,
                "criticality":         round(crit, 3),
                "exposure":            round(exp,  3),
                "patch_compliance":    round(patch, 3),
                "control_coverage":    round(ctrl, 3),
                "confidentiality_imp": round(c_imp, 3),
                "integrity_imp":       round(i_imp, 3),
                "availability_imp":    round(a_imp, 3),
                "asset_criticality_score": round(acs, 3),
                "vuln_count":          n_vulns,
            })

    asset_df = pd.DataFrame(assets)

    # Threat actors
    threat_actors = []
    for ta_type in threat_actor_types:
        cfg = THREAT_ACTORS[ta_type]
        cap_lo, cap_hi = cfg["capability"]
        per_lo, per_hi = cfg["persistence"]
        threat_actors.append({
            "actor_type":     ta_type,
            "capability":     float(rng.uniform(cap_lo, cap_hi) / 10.0),
            "persistence":    float(rng.uniform(per_lo, per_hi) / 10.0),
            "motivation":     cfg["motivation"],
            "n_techniques":   int(rng.integers(3, 12)),
        })
    actor_df = pd.DataFrame(threat_actors)

    return {"assets": asset_df, "actors": actor_df, "seed": seed,
            "n_assets": len(asset_df), "n_actors": len(actor_df)}

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 ENGINE — Threat Scenario Generation + Attack Graph
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_scenarios(env_assets_json, env_actors_json, n_scenarios,
                       selected_vectors, seed, max_path_len):
    """Generate synthetic threat scenarios and build the attack graph."""
    rng  = np.random.default_rng(seed)
    asset_df = pd.read_json(io.StringIO(env_assets_json), orient="records")
    actor_df = pd.read_json(io.StringIO(env_actors_json), orient="records")

    n_assets = len(asset_df)

    # Build attack graph: nodes = assets, edges weighted by difficulty
    G = nx.DiGraph()
    for idx, row in asset_df.iterrows():
        G.add_node(idx, **row.to_dict())
    # Create edges (attack paths between adjacent assets)
    for i in range(n_assets):
        n_targets = min(n_assets - 1, max(1, int(rng.poisson(3))))
        targets   = rng.choice([j for j in range(n_assets) if j != i],
                                size=min(n_targets, n_assets-1), replace=False)
        for j in targets:
            vec    = rng.choice(selected_vectors)
            diff   = ATTACK_VECTORS[vec]["difficulty"]
            cvss_e = ATTACK_VECTORS[vec]["cvss_base"]
            G.add_edge(i, j, vector=vec, difficulty=diff,
                       weight=1.0 - (cvss_e / 10.0), cvss=cvss_e)

    # Sample attack paths for scenarios
    entry_points = asset_df[asset_df["exposure"] > 0.6].index.tolist()
    if not entry_points:
        entry_points = list(range(min(5, n_assets)))
    high_value = asset_df.nlargest(max(3, n_assets//5), "asset_criticality_score").index.tolist()

    rows = []
    for _ in range(n_scenarios):
        actor_row  = actor_df.sample(1, random_state=int(rng.integers(0,9999))).iloc[0]
        asset_row  = asset_df.sample(1, random_state=int(rng.integers(0,9999))).iloc[0]
        vec        = rng.choice(selected_vectors)
        vec_cfg    = ATTACK_VECTORS[vec]

        # CVSS score — NVD-calibrated mixture model
        sev_roll = rng.random()
        if   sev_roll < 0.14: cvss = float(rng.uniform(9.0, 10.0))  # Critical
        elif sev_roll < 0.48: cvss = float(rng.uniform(7.0,  9.0))  # High
        elif sev_roll < 0.98: cvss = float(rng.uniform(4.0,  7.0))  # Medium
        else:                  cvss = float(rng.uniform(0.1,  4.0))  # Low

        exploitability  = float(rng.beta(4, 3))     # 0–1, skewed high
        attack_complexity = float(rng.uniform(0.2, 1.0))
        impact_score    = float((cvss / 10.0) * asset_row["asset_criticality_score"])

        # Shortest path length in attack graph
        src = int(rng.choice(entry_points))
        tgt = int(rng.choice(high_value))
        try:
            path_len = nx.shortest_path_length(G, source=src, target=tgt, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_len = float(rng.integers(2, max_path_len + 1))
        path_len = max(1.0, path_len)

        # Threat likelihood
        threat_likelihood = float(
            actor_row["capability"] * exploitability *
            asset_row["exposure"] * (1 - attack_complexity * 0.3)
        )

        rows.append({
            "actor_type":          actor_row["actor_type"],
            "actor_capability":    float(actor_row["capability"]),
            "actor_persistence":   float(actor_row["persistence"]),
            "asset_type":          asset_row["asset_type"],
            "asset_criticality":   float(asset_row["asset_criticality_score"]),
            "vuln_count":          int(asset_row["vuln_count"]),
            "patch_compliance":    float(asset_row["patch_compliance"]),
            "control_coverage":    float(asset_row["control_coverage"]),
            "exposure":            float(asset_row["exposure"]),
            "attack_vector":       vec,
            "cvss_score":          round(cvss, 2),
            "exploitability":      round(exploitability, 3),
            "attack_complexity":   round(attack_complexity, 3),
            "attack_path_length":  round(path_len, 3),
            "impact_score":        round(impact_score, 3),
            "threat_likelihood":   round(threat_likelihood, 3),
        })

    scenario_df = pd.DataFrame(rows)

    # Derive CVSS severity label
    def severity(s):
        if s >= 9.0: return "Critical"
        if s >= 7.0: return "High"
        if s >= 4.0: return "Medium"
        return "Low"
    scenario_df["cvss_severity"] = scenario_df["cvss_score"].apply(severity)

    graph_data = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
    }
    return scenario_df, graph_data

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 ENGINE — Feature Engineering + Complexity Model
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engineer_features(scenarios_json):
    """Extract and engineer ML-ready features from threat scenarios."""
    df = pd.read_json(io.StringIO(scenarios_json), orient="records")

    feat = pd.DataFrame()
    # F1: Asset Criticality Score (already computed)
    feat["asset_criticality"]        = df["asset_criticality"].clip(0, 1)
    # F2: Log-normalised vulnerability count
    feat["vuln_count_norm"]          = np.log1p(df["vuln_count"]) / np.log1p(df["vuln_count"].max())
    # F3: CVSS weighted average (normalised)
    feat["cvss_weighted_avg"]        = df["cvss_score"] / 10.0
    # F4: Exploitability score
    feat["exploitability_score"]     = df["exploitability"].clip(0, 1)
    # F5: Inverse attack path length (shorter path = higher risk)
    feat["attack_path_length_inv"]   = 1.0 / df["attack_path_length"].clip(lower=0.5)
    feat["attack_path_length_inv"]   = feat["attack_path_length_inv"] / feat["attack_path_length_inv"].max()
    # F6: Threat likelihood
    feat["threat_likelihood"]        = df["threat_likelihood"].clip(0, 1)
    # F7: Network exposure ordinal
    feat["exposure_level"]           = df["exposure"].clip(0, 1)
    # F8: Patch compliance inverse
    feat["patch_compliance_inv"]     = 1.0 - df["patch_compliance"].clip(0, 1)
    # F9: Attacker capability
    feat["attacker_capability"]      = df["actor_capability"].clip(0, 1)
    # F10: Control effectiveness inverse
    feat["control_effectiveness_inv"]= 1.0 - df["control_coverage"].clip(0, 1)

    # ── TARGET: Composite Risk Score 0–10 ──────────────────────────────────
    # Known weighted formula + calibrated noise (allows ground-truth ML validation)
    risk_raw = (
        0.20 * feat["asset_criticality"] * 10 +
        0.15 * feat["vuln_count_norm"]   * 10 +
        0.15 * feat["cvss_weighted_avg"] * 10 +
        0.12 * feat["exploitability_score"] * 10 +
        0.10 * feat["attack_path_length_inv"] * 10 +
        0.10 * feat["threat_likelihood"] * 10 +
        0.08 * feat["exposure_level"]    * 10 +
        0.05 * feat["patch_compliance_inv"] * 10 +
        0.03 * feat["attacker_capability"] * 10 +
        0.02 * feat["control_effectiveness_inv"] * 10
    )
    noise = np.random.default_rng(42).normal(0, 0.4, len(risk_raw))
    feat["risk_score"] = np.clip(risk_raw + noise, 0, 10).round(3)

    def risk_label(s):
        if s >= 7.5: return "Critical"
        if s >= 5.0: return "High"
        if s >= 2.5: return "Medium"
        return "Low"
    feat["risk_label"] = feat["risk_score"].apply(risk_label)

    # Carry over categorical columns for EDA
    feat["actor_type"]   = df["actor_type"].values
    feat["asset_type"]   = df["asset_type"].values
    feat["attack_vector"]= df["attack_vector"].values
    feat["cvss_severity"]= df["cvss_severity"].values

    return feat

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 ENGINE — ML Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_models(features_json, rf_params, gb_params, test_size, cv_folds):
    """Train Random Forest and Gradient Boosting regressors; compute metrics."""
    feat_df = pd.read_json(io.StringIO(features_json), orient="records")

    X = feat_df[FEATURE_NAMES].values
    y = feat_df["risk_score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    results = {}
    for name, model in [
        ("Random Forest",       RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)),
        ("Gradient Boosting",   GradientBoostingRegressor(**gb_params, random_state=42)),
    ]:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = (time.perf_counter() - t1) * 1000  # ms

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(np.mean(np.abs((y_test - y_pred) /
                     np.clip(np.abs(y_test), 1e-9, None))) * 100)

        cv_scores = cross_val_score(model, X, y, cv=cv_folds,
                                    scoring="r2", n_jobs=-1)

        # Feature importance
        perm = permutation_importance(model, X_test, y_test,
                                      n_repeats=5, random_state=42, n_jobs=-1)

        # SHAP (TreeExplainer — fast for tree models)
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test[:200])  # cap for speed

        results[name] = {
            "model":        model,
            "y_test":       y_test.tolist(),
            "y_pred":       y_pred.tolist(),
            "r2":           round(r2, 4),
            "mae":          round(mae, 4),
            "rmse":         round(rmse, 4),
            "mape":         round(mape, 2),
            "cv_r2_mean":   round(float(cv_scores.mean()), 4),
            "cv_r2_std":    round(float(cv_scores.std()),  4),
            "train_time_s": round(train_time, 4),
            "infer_ms":     round(infer_time, 3),
            "perm_importance_mean": perm.importances_mean.tolist(),
            "perm_importance_std":  perm.importances_std.tolist(),
            "shap_values":  shap_vals.tolist(),
            "shap_X":       X_test[:200].tolist(),
            "n_train":      len(X_train),
            "n_test":       len(X_test),
        }

    return results

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 ENGINE — Scalability Benchmarking
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_scalability_benchmark(n_sizes, base_asset_mix_json, base_threat_actors,
                               base_vectors, seed, rf_params, gb_params):
    """Benchmark all 4 pipeline stages across increasing N."""
    asset_mix = json.loads(base_asset_mix_json)
    records   = []

    for n in n_sizes:
        row = {"N": n}

        # ── Stage 1: Data Generation ────────────────────────────────────────
        tracemalloc.start(); t0 = time.perf_counter()
        env = simulate_environment(n, seed, asset_mix, base_threat_actors)
        row["gen_time"]  = round(time.perf_counter() - t0, 5)
        _, row["gen_mem"] = tracemalloc.get_traced_memory(); tracemalloc.stop()
        row["gen_mem"] = round(row["gen_mem"] / 1024, 1)

        # ── Stage 2: Scenario Generation (n_scenarios = n*2) ────────────────
        n_sc = max(100, n * 2)
        assets_json = env["assets"].to_json(orient="records")
        actors_json = env["actors"].to_json(orient="records")
        tracemalloc.start(); t0 = time.perf_counter()
        sc_df, _ = generate_scenarios(assets_json, actors_json, n_sc,
                                       base_vectors, seed, max_path_len=8)
        row["scen_time"]  = round(time.perf_counter() - t0, 5)
        _, row["scen_mem"] = tracemalloc.get_traced_memory(); tracemalloc.stop()
        row["scen_mem"] = round(row["scen_mem"] / 1024, 1)

        # ── Stage 3: Feature Engineering ────────────────────────────────────
        sc_json = sc_df.to_json(orient="records")
        tracemalloc.start(); t0 = time.perf_counter()
        feat_df = engineer_features(sc_json)
        row["feat_time"]  = round(time.perf_counter() - t0, 5)
        _, row["feat_mem"] = tracemalloc.get_traced_memory(); tracemalloc.stop()
        row["feat_mem"] = round(row["feat_mem"] / 1024, 1)

        # ── Stage 4: ML Training + Inference ───────────────────────────────
        feat_json = feat_df[FEATURE_NAMES + ["risk_score"]].to_json(orient="records")
        tracemalloc.start(); t0 = time.perf_counter()
        X = feat_df[FEATURE_NAMES].values
        y = feat_df["risk_score"].values
        X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr); rf.predict(X_te)
        row["ml_time"]  = round(time.perf_counter() - t0, 5)
        _, row["ml_mem"] = tracemalloc.get_traced_memory(); tracemalloc.stop()
        row["ml_mem"] = round(row["ml_mem"] / 1024, 1)

        row["total_time"]  = round(sum([row["gen_time"], row["scen_time"],
                                        row["feat_time"], row["ml_time"]]), 5)
        row["total_mem"]   = round(max(row["gen_mem"], row["scen_mem"],
                                        row["feat_mem"], row["ml_mem"]), 1)
        row["throughput"]  = round(n_sc / row["total_time"] if row["total_time"] > 0 else 0, 1)
        records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 PASTA-ML Controls")
    st.caption("Parameters flow through all 6 steps automatically.")

    st.markdown("### 🏗️ Step 2 — Environment")
    n_assets  = st.slider("Total Assets", 20, 2000, 150, step=10)
    rng_seed  = st.number_input("Random Seed", value=42, step=1)

    st.markdown("**Asset Mix (%)**")
    mix_raw = {}
    mix_cols = st.columns(2)
    asset_list = list(ASSET_TYPES.keys())
    for i, at in enumerate(asset_list):
        col = mix_cols[i % 2]
        mix_raw[at] = col.slider(f"{ASSET_TYPES[at]['icon']} {at[:10]}",
                                  0, 100, [25,15,15,10,15,10,10][i], 5,
                                  key=f"mix_{i}")

    st.markdown("**Threat Actors**")
    selected_actors = st.multiselect("Active actors:",
        list(THREAT_ACTORS.keys()),
        default=["APT Group", "Cybercriminal", "Insider Threat"])

    st.divider()
    st.markdown("### ⚔️ Step 3 — Scenarios")
    n_scenarios  = st.slider("Scenarios to generate", 200, 10000, 1000, 100)
    selected_vecs = st.multiselect("Attack Vectors:",
        list(ATTACK_VECTORS.keys()),
        default=list(ATTACK_VECTORS.keys())[:7])
    max_path_len = st.slider("Max Attack Path Length", 2, 15, 6)

    st.divider()
    st.markdown("### 🤖 Step 5 — ML Settings")
    test_size = st.slider("Test Split (%)", 10, 40, 20) / 100
    cv_folds  = st.slider("Cross-Val Folds", 3, 10, 5)

    st.markdown("**Random Forest**")
    rf_n_est  = st.slider("n_estimators (RF)", 50, 500, 150, 50)
    rf_depth  = st.slider("max_depth (RF)", 3, 30, 15)

    st.markdown("**Gradient Boosting**")
    gb_n_est  = st.slider("n_estimators (GB)", 50, 300, 100, 50)
    gb_lr     = st.slider("learning_rate (GB)", 0.01, 0.30, 0.10, 0.01)
    gb_depth  = st.slider("max_depth (GB)", 2, 8, 4)

    st.divider()
    st.markdown("### ⚡ Step 6 — Benchmarks")
    bench_max = st.number_input("Max N for benchmark", value=500, step=50,
                                 min_value=50, max_value=2000)
    bench_pts = st.slider("N points", 4, 12, 7)
    bench_scale = st.radio("N spacing", ["Linear","Log"], horizontal=True)

# Collect params
rf_params = {"n_estimators": rf_n_est, "max_depth": rf_depth,
             "min_samples_leaf": 3, "oob_score": True}
gb_params = {"n_estimators": gb_n_est, "learning_rate": gb_lr,
             "max_depth": gb_depth, "subsample": 0.8}
asset_mix = {k: v for k, v in mix_raw.items() if v > 0}
if not asset_mix:
    asset_mix = {"Cloud VM": 25, "Enterprise App": 25, "Database Server": 25, "Endpoint": 25}
if not selected_actors:
    selected_actors = ["Cybercriminal"]
if not selected_vecs:
    selected_vecs = list(ATTACK_VECTORS.keys())[:5]

bench_sizes = tuple(int(x) for x in (
    np.geomspace(20, bench_max, bench_pts)
    if bench_scale == "Log"
    else np.linspace(20, bench_max, bench_pts, dtype=int)))

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 PASTA-ML Research Framework")
st.markdown(
    "**A Scalable Machine Learning-Integrated Threat Modeling Framework "
    "for Large-Scale Cyber-Physical Systems**  \n"
    "Interactive research pipeline • Academic evaluation tool • 6-step methodology"
)

# Pipeline status badges
ph1_col, ph2_col, ph3_col = st.columns(3)
with ph1_col:
    st.markdown(
        "<span class='phase-badge' style='background:#1a73e8;color:white;'>Phase 1</span>"
        " Step 1: Framework Design &nbsp;|&nbsp; Step 2: Environment Simulation",
        unsafe_allow_html=True)
with ph2_col:
    st.markdown(
        "<span class='phase-badge' style='background:#34A853;color:white;'>Phase 2</span>"
        " Step 3: Scenario Generation &nbsp;|&nbsp; Step 4: Feature Engineering",
        unsafe_allow_html=True)
with ph3_col:
    st.markdown(
        "<span class='phase-badge' style='background:#EA4335;color:white;'>Phase 3</span>"
        " Step 5: ML Risk Estimation &nbsp;|&nbsp; Step 6: Scalability Evaluation",
        unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
(tab_overview,
 tab_step1, tab_step2, tab_step3,
 tab_step4, tab_step5, tab_step6,
 tab_export) = st.tabs([
    "🏠 Overview",
    "📐 Step 1 · Framework",
    "🏗️ Step 2 · Environment",
    "🎲 Step 3 · Scenarios",
    "🔧 Step 4 · Features",
    "🤖 Step 5 · ML Models",
    "⚡ Step 6 · Scalability",
    "📤 Export",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("🏠 Research Pipeline Overview")

    # Pipeline flowchart using Plotly
    steps = [
        ("Phase 1","Step 1","Modified PASTA\nFramework Design","#1a5276"),
        ("Phase 1","Step 2","System Modeling &\nEnvironment Sim.","#1f618d"),
        ("Phase 2","Step 3","Synthetic Threat\nScenario Generation","#17a589"),
        ("Phase 2","Step 4","Feature Engineering\n& Complexity Model","#d68910"),
        ("Phase 3","Step 5","ML-Based\nRisk Estimation","#8e44ad"),
        ("Phase 3","Step 6","Scalability &\nPerformance Eval.","#c0392b"),
    ]
    fig_flow = go.Figure()
    for i, (phase, step, label, color) in enumerate(steps):
        fig_flow.add_trace(go.Scatter(
            x=[i], y=[0],
            mode="markers+text",
            marker=dict(size=70, color=color, line=dict(color="white", width=3)),
            text=[f"<b>{step}</b>"],
            textposition="middle center",
            textfont=dict(color="white", size=11),
            hovertemplate=f"<b>{phase} | {step}</b><br>{label.replace(chr(10),' ')}<extra></extra>",
            name=f"{step}: {label.replace(chr(10),' ')}",
            showlegend=False,
        ))
        if i > 0:
            fig_flow.add_annotation(
                x=i-0.42, y=0, ax=i-0.58, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowcolor="#aaaaaa", arrowwidth=2,
            )
        fig_flow.add_annotation(
            x=i, y=-0.18,
            text=f"<b style='color:{color};'>{phase}</b><br><span style='font-size:10px'>{label}</span>",
            showarrow=False, font=dict(size=10), align="center",
        )

    fig_flow.update_layout(
        height=220, margin=dict(l=20,r=20,t=20,b=70),
        xaxis=dict(visible=False, range=[-0.5, len(steps)-0.5]),
        yaxis=dict(visible=False, range=[-0.45, 0.25]),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_flow, use_container_width=True)

    # Step descriptions
    step_info = [
        ("📐 Step 1", "Modified PASTA Framework Design",
         "Phase 1", "#1a73e8",
         "Restructures the traditional 7-stage PASTA methodology to support automated "
         "asset mapping, vulnerability aggregation, and threat vector modeling at scale. "
         "Defines typed data structures for assets, vulnerabilities, and threat actors "
         "that feed downstream ML processing.",
         ["Automated asset mapping", "Scalable pipeline design",
          "Structured data representations", "Modified 7-stage PASTA"]),

        ("🏗️ Step 2", "System Modeling & Threat Environment Simulation",
         "Phase 1", "#1a73e8",
         "Simulates large-scale cyber-physical infrastructure including cloud VMs, IoT "
         "devices, SCADA/ICS, databases, and enterprise applications. Each asset carries "
         "CIA impact ratings, CVSS-calibrated vulnerability counts, patch compliance, "
         "and network exposure attributes.",
         ["7 asset types", "NVD-calibrated vuln distributions",
          "6 threat actor profiles", "Realistic CIA impact ratings"]),

        ("🎲 Step 3", "Synthetic Threat Scenario Generation",
         "Phase 2", "#34A853",
         "Generates large sets of attack scenarios combining assets, vulnerabilities, "
         "and threat vectors. Uses NetworkX for attack graph construction and Dijkstra-"
         "based path analysis. CVSS scores follow the NVD severity distribution "
         "(Critical 14%, High 34%, Medium 50%, Low 2%).",
         ["NetworkX attack graph", "12 attack vectors",
          "NVD-calibrated CVSS", "Dijkstra path analysis"]),

        ("🔧 Step 4", "Feature Engineering & Complexity Characterization",
         "Phase 2", "#34A853",
         "Transforms raw threat scenarios into 10 engineered ML-ready features: asset "
         "criticality score, log-normalised vulnerability count, CVSS weighted average, "
         "exploitability, inverse attack path length, threat likelihood, exposure level, "
         "patch compliance inverse, attacker capability, and control effectiveness inverse.",
         ["10 engineered features", "Complexity characterization",
          "Known risk formula target", "Correlation analysis"]),

        ("🤖 Step 5", "ML-Based Risk Estimation",
         "Phase 3", "#EA4335",
         "Trains Random Forest and Gradient Boosting regressors to predict the composite "
         "risk score (0–10). Evaluates with R², MAE, RMSE, MAPE, and k-fold "
         "cross-validation. Produces SHAP explainability plots and permutation-based "
         "feature importance for academic transparency.",
         ["Random Forest + Gradient Boosting", "SHAP explainability",
          "k-fold cross-validation", "Permutation importance"]),

        ("⚡ Step 6", "Scalability & Performance Evaluation",
         "Phase 3", "#EA4335",
         "Benchmarks all 4 pipeline stages (data generation, scenario generation, "
         "feature engineering, ML training+inference) as problem size N scales from "
         "small to large. Reports wall-clock time, peak memory (KB), throughput "
         "(scenarios/s), and empirical O(N^k) complexity from log-log fit.",
         ["4-stage pipeline benchmarking", "Wall-clock + memory profiling",
          "Throughput measurement", "O(N^k) complexity fit"]),
    ]

    cols = st.columns(2)
    for i, (label, title, phase, pcolor, desc, highlights) in enumerate(step_info):
        with cols[i % 2]:
            badges = " ".join(
                f"<span class='metric-pill'>{h}</span>" for h in highlights)
            st.markdown(
                f"<div class='step-card'>"
                f"<b style='color:{pcolor};'>{label}: {title}</b><br>"
                f"<small style='color:#666;'>{phase}</small><br><br>"
                f"{desc}<br><br>{badges}"
                f"</div>",
                unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📖 Research Motivation")
    st.markdown("""
    <div class='callout-info'>
    <b>Problem Statement:</b> As modern cyber-physical and distributed systems grow in complexity,
    the PASTA threat modeling framework faces quantifiable scalability constraints.
    Without explicit scalability measurement, it remains unclear how PASTA-based pipelines
    perform as the number of assets, vulnerabilities, and attack scenarios increases.
    <br><br>
    <b>Proposed Solution:</b> PASTA-ML extends PASTA with synthetic data generation,
    ML-based risk estimation, and rigorous empirical scalability evaluation — producing
    measurable metrics (computation time, memory usage, throughput) across all pipeline stages.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB: STEP 1 — Framework Design
# ═══════════════════════════════════════════════════════════════════════════
with tab_step1:
    st.subheader("📐 Step 1: Modified PASTA Framework Design")
    st.markdown(
        "<div class='callout-info'>This step restructures the traditional PASTA methodology "
        "into an automated, scalable pipeline. Each of the 7 PASTA stages is mapped to "
        "model variables and ML pipeline components.</div>", unsafe_allow_html=True)

    # 7-stage interactive viewer
    st.markdown("#### 🗺️ Modified PASTA — 7-Stage Pipeline")
    stage_sel = st.select_slider("Select Stage",
        options=[f"{PASTA_STAGES[s]['icon']} Stage {s}: {PASTA_STAGES[s]['name']}"
                 for s in PASTA_STAGES],
        value=f"{PASTA_STAGES[1]['icon']} Stage 1: {PASTA_STAGES[1]['name']}")
    stage_num = int(stage_sel.split("Stage ")[1].split(":")[0])

    STAGE_DETAIL = {
        1: {"vars": ["OrgMaturity", "Risk Appetite"],
            "modification": "Automated objective extraction from compliance templates (HIPAA, GDPR, PCI-DSS). "
                            "Structured BusinessObjective records replace manual workshops.",
            "scalability": "O(1) per system — does not scale with asset count. "
                           "Bottleneck is stakeholder time, not computation.",
            "ml_link": "Defines the risk appetite threshold used to label ML predictions as "
                       "acceptable / unacceptable risk.",
            "formula": "Risk_Appetite_Score = weighted_avg(Compliance_Requirements)"},
        2: {"vars": ["AssetsCount", "AssetValue", "ExposureLevel"],
            "modification": "Automated asset discovery via CMDB integration and network scanning APIs. "
                            "Each asset is typed and attributed automatically.",
            "scalability": "O(N) storage. O(N²) for dependency mapping in dense architectures. "
                           "Graph-based representation reduces to O(N·avg_degree) with sparse graphs.",
            "ml_link": "Asset criticality score (ACS = CIA_weighted × exposure) is Feature F1 in the ML model.",
            "formula": "ACS = (0.4×C_imp + 0.3×I_imp + 0.3×A_imp) × exposure_factor"},
        3: {"vars": ["Complexity", "ChangeRate", "DataFlows"],
            "modification": "Automated DFD generation from architectural descriptions. "
                            "Trust boundary extraction from network segmentation rules.",
            "scalability": "DFD complexity = O(N²) worst-case for fully-connected systems. "
                           "Sparse enterprise architectures: O(N·log N).",
            "ml_link": "System complexity and change rate contribute to attack path enumeration "
                       "cost in the NetworkX graph used for Feature F5.",
            "formula": "Complexity_Score = nodes × avg_connectivity × change_frequency"},
        4: {"vars": ["ThreatVectors", "ThreatActors", "T_weight"],
            "modification": "MITRE ATT&CK and ENISA integration for automated threat enumeration. "
                            "Threat actor profiling using STIX 2.1 vocabulary.",
            "scalability": "O(A×T) combinatorial expansion. Pruning via actor capability thresholds "
                           "reduces to O(A×T×P_exploit > threshold).",
            "ml_link": "Threat likelihood (Feature F6) and attacker capability (Feature F9) "
                       "derive directly from threat actor profiles.",
            "formula": "ThreatLikelihood = capability × exploitability × exposure × (1 − controls×0.3)"},
        5: {"vars": ["VulnCount", "CVSSScore", "ExploitAvailability"],
            "modification": "NVD/CVE integration for automated vulnerability enumeration. "
                            "EPSS-weighted exploitability scoring supplements CVSS base scores.",
            "scalability": "O(V×A) CVE lookups. Dominant bottleneck for large-scale systems. "
                           "Addressed by batched API calls and local caching.",
            "ml_link": "CVSS weighted average (F3), exploitability score (F4), "
                       "log-normalised vuln count (F2), and patch compliance inverse (F8).",
            "formula": "VES = CVSS_base × temporal_modifier × exploit_availability_weight"},
        6: {"vars": ["AttackPaths", "AttackTrees", "PathLength"],
            "modification": "NetworkX-based attack graph replaces manual attack trees. "
                            "Dijkstra (easiest path) and K-shortest paths algorithms enumerate attack chains.",
            "scalability": "NP-hard in general. Practical O(N²·log N) with Dijkstra on sparse graphs. "
                           "Depth-limited BFS controls exponential growth.",
            "ml_link": "Inverse shortest attack path length (Feature F5). "
                       "Shorter paths → higher risk → higher feature value.",
            "formula": "PathRisk = P(path) × target_criticality; P = ∏P(step_i)"},
        7: {"vars": ["RiskScore", "BusinessImpact", "Mitigation"],
            "modification": "ML model outputs replace manual risk matrices. "
                            "Automated countermeasure prioritisation via risk-delta ranking.",
            "scalability": "O(N) aggregation — the only stage that scales linearly by design. "
                           "ML inference is the main cost: ~milliseconds per scenario.",
            "ml_link": "The composite Risk Score (0–10) is the ML model's target variable. "
                       "Predicted scores feed automated countermeasure prioritisation.",
            "formula": "Risk = 0.20·ACS + 0.15·VulnNorm + 0.15·CVSS + 0.12·Exploit + "
                       "0.10·PathInv + 0.10·ThreatLH + 0.08·Exposure + ..."},
    }

    sd = STAGE_DETAIL[stage_num]
    si = PASTA_STAGES[stage_num]
    st.markdown(
        f"<div style='background:{si['color']};padding:16px 20px;border-radius:10px;"
        f"color:white;margin-bottom:12px;'>"
        f"<h3 style='margin:0;color:white;'>{si['icon']} Stage {stage_num}: {si['name']}</h3>"
        f"</div>", unsafe_allow_html=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("##### 🔧 Modification from Traditional PASTA")
        st.markdown(f"<div class='callout-info'>{sd['modification']}</div>", unsafe_allow_html=True)
        st.markdown("##### ⚠️ Scalability Analysis")
        st.markdown(f"<div class='callout-warn'>{sd['scalability']}</div>", unsafe_allow_html=True)
    with dc2:
        st.markdown("##### 🤖 ML Pipeline Link")
        st.markdown(f"<div class='callout-good'>{sd['ml_link']}</div>", unsafe_allow_html=True)
        st.markdown("##### 📐 Key Formula")
        st.markdown(f"<div class='formula-box'>{sd['formula']}</div>", unsafe_allow_html=True)
        st.markdown("##### 📌 Variables")
        for v in sd["vars"]:
            st.markdown(f"  `{v}`", unsafe_allow_html=True)

    # All-stages scalability table
    st.divider()
    st.markdown("#### 📋 All Stages — Scalability Summary")
    scalability_table = pd.DataFrame([
        {"Stage": f"{PASTA_STAGES[s]['icon']} {s}. {PASTA_STAGES[s]['name']}",
         "Complexity Class": c,
         "Primary Driver": d,
         "PASTA-ML Mitigation": m}
        for s, c, d, m in [
            (1, "O(1)",      "Stakeholder time",         "Compliance template automation"),
            (2, "O(N)",      "Asset count",              "CMDB / discovery API integration"),
            (3, "O(N²)",     "Graph density",            "Sparse graph + hierarchical decomposition"),
            (4, "O(A×T)",    "Threat combinations",      "MITRE ATT&CK pruning by capability"),
            (5, "O(V×A)",    "CVE lookup volume",        "Batched NVD API + local CVSS cache"),
            (6, "NP-hard",   "Attack path enumeration",  "Depth-limited Dijkstra + K-shortest paths"),
            (7, "O(N)",      "Scenario scoring",         "Vectorised ML inference (batch predict)"),
        ]
    ])
    st.dataframe(scalability_table, use_container_width=True, hide_index=True)

    # Complexity growth chart
    st.markdown("#### 📈 Theoretical Complexity Growth by Stage")
    N_vals = np.arange(10, 1001, 10)
    comp_data = {
        "Stage 1 – O(1)":     np.ones_like(N_vals) * 1.0,
        "Stage 2 – O(N)":     N_vals.astype(float),
        "Stage 3 – O(N²)":    N_vals.astype(float) ** 2,
        "Stage 4 – O(A×T)":   N_vals.astype(float) * np.log(N_vals),
        "Stage 5 – O(V×A)":   N_vals.astype(float) ** 1.5,
        "Stage 6 – O(NP)":    2.0 ** (N_vals / 50.0),
        "Stage 7 – O(N)":     N_vals.astype(float) * 0.01,
    }
    # Normalise to [0,1] for comparison
    fig_comp = go.Figure()
    colors_c = ["#1a5276","#1f618d","#2874a6","#17a589","#d68910","#ba4a00","#7d3c98"]
    for (label, vals), col in zip(comp_data.items(), colors_c):
        norm = vals / vals.max()
        fig_comp.add_trace(go.Scatter(
            x=N_vals, y=norm, mode="lines", name=label,
            line=dict(color=col, width=2)))
    fig_comp.update_layout(
        title="Normalised Complexity Growth — PASTA-ML Stages vs. Asset Count N",
        xaxis_title="N (Assets / Problem Size)",
        yaxis_title="Normalised Computational Cost",
        height=380, margin=dict(l=0,r=0,t=40,b=0),
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
        yaxis_type="log",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB: STEP 2 — Environment Simulation
# ═══════════════════════════════════════════════════════════════════════════
with tab_step2:
    st.subheader("🏗️ Step 2: System Modeling & Threat Environment Simulation")
    st.markdown(
        "<div class='callout-info'>Simulates a large-scale cyber-physical infrastructure "
        "with configurable asset types, vulnerability distributions, and threat actor profiles. "
        "All parameters are controlled from the sidebar.</div>", unsafe_allow_html=True)

    if st.button("▶ Run Environment Simulation", type="primary", key="run_env"):
        with st.spinner("Simulating environment…"):
            env = simulate_environment(n_assets, rng_seed, asset_mix, selected_actors)
            st.session_state["env"] = env

    if st.session_state["env"] is None:
        st.info("👆 Click **Run Environment Simulation** to build the system model.")
    else:
        env = st.session_state["env"]
        asset_df = env["assets"]
        actor_df = env["actors"]

        # KPIs
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total Assets",        f"{env['n_assets']:,}")
        k2.metric("Threat Actors",       env['n_actors'])
        k3.metric("Avg Criticality",     f"{asset_df['asset_criticality_score'].mean():.3f}")
        k4.metric("Avg Vulnerabilities", f"{asset_df['vuln_count'].mean():.1f}")
        k5.metric("Avg Exposure",        f"{asset_df['exposure'].mean():.3f}")

        # Asset type distribution
        ec1, ec2 = st.columns(2)
        with ec1:
            type_counts = asset_df["asset_type"].value_counts().reset_index()
            type_counts.columns = ["Asset Type","Count"]
            fig_pie = px.pie(type_counts, values="Count", names="Asset Type",
                             title="Asset Type Distribution", hole=0.4,
                             color_discrete_sequence=[ASSET_TYPES[t]["color"]
                                                      for t in type_counts["Asset Type"]])
            fig_pie.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        with ec2:
            fig_exp = px.box(asset_df, x="asset_type", y="exposure",
                             color="asset_type",
                             color_discrete_map={t: ASSET_TYPES[t]["color"] for t in ASSET_TYPES},
                             title="Exposure Distribution by Asset Type")
            fig_exp.update_xaxes(tickangle=30)
            fig_exp.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
            st.plotly_chart(fig_exp, use_container_width=True)

        # Criticality vs Vulnerability scatter
        fig_scatter = px.scatter(
            asset_df, x="asset_criticality_score", y="vuln_count",
            color="asset_type", size="exposure",
            color_discrete_map={t: ASSET_TYPES[t]["color"] for t in ASSET_TYPES},
            hover_data=["asset_id", "patch_compliance", "control_coverage"],
            title="Asset Criticality vs. Vulnerability Count (bubble = exposure)",
            labels={"asset_criticality_score":"Asset Criticality Score (ACS)",
                    "vuln_count":"Vulnerability Count"})
        fig_scatter.update_layout(height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation heatmap
        corr_cols = ["asset_criticality_score","vuln_count","exposure",
                     "patch_compliance","control_coverage",
                     "confidentiality_imp","integrity_imp","availability_imp"]
        corr_m = asset_df[corr_cols].corr()
        fig_heat = px.imshow(corr_m, text_auto=".2f", aspect="auto",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Asset Attribute Correlation Matrix")
        fig_heat.update_layout(height=380, margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Threat actor profiles
        st.markdown("#### 🎯 Threat Actor Profiles")
        if not actor_df.empty:
            fig_actors = px.bar(
                actor_df.melt(id_vars=["actor_type","motivation"],
                              value_vars=["capability","persistence"],
                              var_name="Metric", value_name="Score"),
                x="Score", y="actor_type", color="Metric",
                barmode="group", orientation="h",
                color_discrete_map={"capability":"#c0392b","persistence":"#2980b9"},
                title="Threat Actor Capability & Persistence Scores")
            fig_actors.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_actors, use_container_width=True)
            st.dataframe(actor_df, use_container_width=True, hide_index=True)

        # Asset data preview
        st.markdown("#### 📋 Asset Inventory (first 25 rows)")
        st.dataframe(asset_df.head(25), use_container_width=True)
        st.download_button("📥 Download Asset Inventory (CSV)",
                           asset_df.to_csv(index=False).encode(),
                           "asset_inventory.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# TAB: STEP 3 — Scenario Generation
# ═══════════════════════════════════════════════════════════════════════════
with tab_step3:
    st.subheader("🎲 Step 3: Synthetic Threat Scenario Generation")
    st.markdown(
        "<div class='callout-info'>Generates attack scenarios combining assets, "
        "vulnerabilities, and threat vectors. Builds a NetworkX attack graph and "
        "computes shortest attack paths using Dijkstra's algorithm. "
        "CVSS scores follow NVD 2024 severity distribution.</div>",
        unsafe_allow_html=True)

    env_ready = st.session_state["env"] is not None
    if not env_ready:
        st.warning("⚠️ Please run **Step 2 — Environment Simulation** first.")
    else:
        if st.button("▶ Generate Threat Scenarios", type="primary", key="run_scen"):
            env = st.session_state["env"]
            with st.spinner(f"Generating {n_scenarios:,} threat scenarios + attack graph…"):
                sc_df, g_data = generate_scenarios(
                    env["assets"].to_json(orient="records"),
                    env["actors"].to_json(orient="records"),
                    n_scenarios, tuple(selected_vecs), rng_seed, max_path_len)
                st.session_state["scenarios"] = sc_df
                st.session_state["attack_graph"] = g_data

        if st.session_state["scenarios"] is None:
            st.info("👆 Click **Generate Threat Scenarios** to proceed.")
        else:
            sc_df  = st.session_state["scenarios"]
            g_data = st.session_state["attack_graph"]

            # KPIs
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Scenarios",       f"{len(sc_df):,}")
            k2.metric("Graph Nodes",     g_data["n_nodes"])
            k3.metric("Graph Edges",     g_data["n_edges"])
            k4.metric("Graph Density",   f"{g_data['density']:.4f}")
            k5.metric("Avg CVSS",        f"{sc_df['cvss_score'].mean():.2f}")

            sc1, sc2 = st.columns(2)
            with sc1:
                sev_counts = sc_df["cvss_severity"].value_counts().reset_index()
                sev_counts.columns = ["Severity","Count"]
                sev_order = ["Critical","High","Medium","Low"]
                sev_color = {"Critical":"#c0392b","High":"#e67e22",
                             "Medium":"#f1c40f","Low":"#27ae60"}
                fig_sev = px.bar(
                    sev_counts[sev_counts["Severity"].isin(sev_order)].sort_values(
                        "Severity", key=lambda x: x.map({s:i for i,s in enumerate(sev_order)})),
                    x="Severity", y="Count",
                    color="Severity", color_discrete_map=sev_color,
                    title="Scenario Distribution by CVSS Severity (NVD 2024 calibrated)")
                fig_sev.update_layout(height=310, margin=dict(l=0,r=0,t=40,b=0),
                                      showlegend=False)
                st.plotly_chart(fig_sev, use_container_width=True)

            with sc2:
                fig_cvss = px.histogram(sc_df, x="cvss_score", nbins=40,
                    color_discrete_sequence=["#2980b9"],
                    title="CVSS Score Distribution (target: mean ≈ 6.5–7.2)")
                fig_cvss.add_vline(x=sc_df["cvss_score"].mean(), line_dash="dash",
                    annotation_text=f"Mean={sc_df['cvss_score'].mean():.2f}",
                    annotation_position="top right")
                fig_cvss.update_layout(height=310, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_cvss, use_container_width=True)

            # Attack vector usage
            vec_counts = sc_df["attack_vector"].value_counts().reset_index()
            vec_counts.columns = ["Attack Vector","Count"]
            vec_counts["Difficulty"] = vec_counts["Attack Vector"].map(
                lambda v: ATTACK_VECTORS.get(v,{}).get("difficulty", 0.5))
            fig_vec = px.bar(vec_counts.sort_values("Count", ascending=True),
                             x="Count", y="Attack Vector", orientation="h",
                             color="Difficulty",
                             color_continuous_scale="RdYlGn_r",
                             title="Attack Vector Frequency (color = difficulty)")
            fig_vec.update_layout(height=340, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_vec, use_container_width=True)

            # Threat likelihood vs CVSS vs path length
            fig_3d = px.scatter(
                sc_df.sample(min(500, len(sc_df)), random_state=42),
                x="cvss_score", y="threat_likelihood",
                color="actor_type",
                size="attack_path_length",
                hover_data=["asset_type","attack_vector"],
                title="CVSS Score vs Threat Likelihood (size = path length, color = actor)",
            )
            fig_3d.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_3d, use_container_width=True)

            # Attack path distribution
            sc3, sc4 = st.columns(2)
            with sc3:
                fig_path = px.histogram(sc_df, x="attack_path_length", nbins=30,
                    color_discrete_sequence=["#8e44ad"],
                    title="Attack Path Length Distribution")
                fig_path.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_path, use_container_width=True)
            with sc4:
                actor_sev = sc_df.groupby(["actor_type","cvss_severity"])["cvss_score"]\
                    .count().reset_index()
                actor_sev.columns = ["Actor","Severity","Count"]
                fig_as = px.bar(actor_sev, x="Actor", y="Count", color="Severity",
                    color_discrete_map=sev_color, barmode="stack",
                    title="Severity Mix by Threat Actor")
                fig_as.update_xaxes(tickangle=25)
                fig_as.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_as, use_container_width=True)

            st.markdown("#### 📋 Scenario Dataset (first 20)")
            st.dataframe(sc_df.head(20), use_container_width=True)
            st.download_button("📥 Download Scenarios (CSV)",
                               sc_df.to_csv(index=False).encode(),
                               "threat_scenarios.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# TAB: STEP 4 — Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════
with tab_step4:
    st.subheader("🔧 Step 4: Feature Engineering & Complexity Characterization")
    st.markdown(
        "<div class='callout-info'>Transforms raw threat scenarios into 10 engineered "
        "features and derives the composite risk score target variable using a known "
        "weighted formula — enabling ground-truth ML validation.</div>",
        unsafe_allow_html=True)

    scen_ready = st.session_state["scenarios"] is not None
    if not scen_ready:
        st.warning("⚠️ Please complete **Step 3 — Scenario Generation** first.")
    else:
        if st.button("▶ Engineer Features", type="primary", key="run_feat"):
            with st.spinner("Engineering features…"):
                feat_df = engineer_features(
                    st.session_state["scenarios"].to_json(orient="records"))
                st.session_state["features"] = feat_df

        if st.session_state["features"] is None:
            st.info("👆 Click **Engineer Features** to proceed.")
        else:
            feat_df = st.session_state["features"]

            # KPIs
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Features Engineered", len(FEATURE_NAMES))
            k2.metric("Scenarios",           f"{len(feat_df):,}")
            k3.metric("Mean Risk Score",      f"{feat_df['risk_score'].mean():.2f}")
            k4.metric("Risk Score Std",       f"{feat_df['risk_score'].std():.2f}")

            # Feature descriptions
            st.markdown("#### 📐 Engineered Feature Catalogue")
            feat_info = pd.DataFrame([
                {"Feature": f, "Description": FEATURE_DESCRIPTIONS[f],
                 "Mean":    round(feat_df[f].mean(), 3),
                 "Std":     round(feat_df[f].std(),  3),
                 "Min":     round(feat_df[f].min(),  3),
                 "Max":     round(feat_df[f].max(),  3)}
                for f in FEATURE_NAMES
            ])
            st.dataframe(feat_info, use_container_width=True, hide_index=True)

            # Risk score distribution
            fc1, fc2 = st.columns(2)
            with fc1:
                fig_risk = px.histogram(feat_df, x="risk_score", nbins=40,
                    color_discrete_sequence=["#e74c3c"],
                    title="Target Variable: Risk Score Distribution (0–10)")
                fig_risk.add_vline(x=feat_df["risk_score"].mean(), line_dash="dash",
                    annotation_text=f"Mean={feat_df['risk_score'].mean():.2f}",
                    annotation_position="top right")
                fig_risk.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_risk, use_container_width=True)

            with fc2:
                risk_lbl_cnt = feat_df["risk_label"].value_counts().reset_index()
                risk_lbl_cnt.columns = ["Risk Level","Count"]
                fig_rlbl = px.pie(risk_lbl_cnt, values="Count", names="Risk Level",
                    hole=0.42,
                    color_discrete_map={"Critical":"#c0392b","High":"#e67e22",
                                        "Medium":"#f1c40f","Low":"#27ae60"},
                    title="Risk Level Distribution")
                fig_rlbl.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_rlbl, use_container_width=True)

            # Feature correlation with risk_score
            st.markdown("#### 🔗 Feature–Risk Correlation")
            corr_risk = feat_df[FEATURE_NAMES + ["risk_score"]].corr()["risk_score"]\
                .drop("risk_score").sort_values(ascending=False)
            fig_corr = px.bar(
                x=corr_risk.values, y=corr_risk.index, orientation="h",
                color=corr_risk.values,
                color_continuous_scale="RdBu", range_color=[-1,1],
                title="Pearson Correlation of Each Feature with Risk Score",
                labels={"x":"Correlation","y":"Feature"})
            fig_corr.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0),
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Full correlation heatmap
            corr_all = feat_df[FEATURE_NAMES + ["risk_score"]].corr()
            fig_cheat = px.imshow(corr_all, text_auto=".2f", aspect="auto",
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  title="Full Feature Correlation Heatmap")
            fig_cheat.update_layout(height=480, margin=dict(l=0,r=0,t=50,b=0))
            st.plotly_chart(fig_cheat, use_container_width=True)

            # Complexity characterisation model
            st.markdown("#### 📊 Complexity Characterization — Scenario Count vs. Feature Computation Time")
            st.caption("Empirical measurement of feature engineering cost as N grows.")
            n_vals_c = [100, 250, 500, 1000, 2000,
                        min(5000, len(feat_df))]
            times_c  = []
            sc_json  = st.session_state["scenarios"].to_json(orient="records")
            sc_full  = pd.read_json(io.StringIO(sc_json), orient="records")
            for nv in n_vals_c:
                if nv > len(sc_full): break
                t0 = time.perf_counter()
                engineer_features(sc_full.head(nv).to_json(orient="records"))
                times_c.append(round(time.perf_counter() - t0, 5))
                n_vals_c_used = n_vals_c[:len(times_c)]

            fig_cc = px.line(x=n_vals_c_used, y=times_c, markers=True,
                color_discrete_sequence=["#17a589"],
                labels={"x":"N (Scenarios)","y":"Feature Engineering Time (s)"},
                title="Feature Engineering Time vs. Scenario Count")
            fig_cc.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_cc, use_container_width=True)

            # Risk score formula breakdown
            st.markdown("#### 📐 Risk Score Formula (Ground-Truth Target)")
            st.markdown("""
<div class='formula-box'>
Risk_Score = 0.20·ACS×10 + 0.15·VulnNorm×10 + 0.15·CVSS_avg×10<br>
           + 0.12·Exploit×10 + 0.10·PathInv×10 + 0.10·ThreatLH×10<br>
           + 0.08·Exposure×10 + 0.05·PatchInv×10<br>
           + 0.03·Capability×10 + 0.02·CtrlInv×10<br>
           + N(0, 0.4)  [calibrated noise]<br>
           → clipped to [0, 10]
</div>
""", unsafe_allow_html=True)

            st.download_button("📥 Download Feature Dataset (CSV)",
                               feat_df.to_csv(index=False).encode(),
                               "engineered_features.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# TAB: STEP 5 — ML Risk Estimation
# ═══════════════════════════════════════════════════════════════════════════
with tab_step5:
    st.subheader("🤖 Step 5: Machine Learning-Based Risk Estimation")
    st.markdown(
        "<div class='callout-info'>Trains Random Forest and Gradient Boosting regressors "
        "on the engineered features to predict the composite risk score. Evaluates with "
        "R², MAE, RMSE, MAPE, k-fold cross-validation, SHAP explainability, and "
        "permutation-based feature importance.</div>", unsafe_allow_html=True)

    feat_ready = st.session_state["features"] is not None
    if not feat_ready:
        st.warning("⚠️ Please complete **Step 4 — Feature Engineering** first.")
    else:
        if st.button("▶ Train & Evaluate ML Models", type="primary", key="run_ml"):
            feat_df = st.session_state["features"]
            feat_json = feat_df[FEATURE_NAMES + ["risk_score"]].to_json(orient="records")
            with st.spinner("Training Random Forest and Gradient Boosting… computing SHAP…"):
                ml_res = train_models(feat_json, rf_params, gb_params,
                                      test_size, cv_folds)
                st.session_state["ml_results"] = ml_res

        if st.session_state["ml_results"] is None:
            st.info("👆 Click **Train & Evaluate ML Models** to proceed.")
        else:
            ml_res = st.session_state["ml_results"]

            # Model comparison metrics table
            st.markdown("#### 📊 Model Comparison")
            metric_rows = []
            for mname, res in ml_res.items():
                metric_rows.append({
                    "Model": mname,
                    "R²": res["r2"],
                    "MAE": res["mae"],
                    "RMSE": res["rmse"],
                    "MAPE (%)": res["mape"],
                    f"CV R² ({cv_folds}-fold)": f"{res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}",
                    "Train Time (s)": res["train_time_s"],
                    "Infer Time (ms)": res["infer_ms"],
                    "Train N": res["n_train"],
                    "Test N": res["n_test"],
                })
            mdf = pd.DataFrame(metric_rows)
            st.dataframe(mdf.set_index("Model"), use_container_width=True)

            # Performance badge
            best_r2 = max(ml_res[m]["r2"] for m in ml_res)
            if best_r2 >= 0.90:
                badge, bcol = "🟢 Excellent (R² ≥ 0.90)", "#27ae60"
            elif best_r2 >= 0.85:
                badge, bcol = "🟡 Good (R² ≥ 0.85)", "#f39c12"
            else:
                badge, bcol = "🔴 Needs tuning (R² < 0.85)", "#c0392b"
            st.markdown(
                f"<div class='callout-good'>Best R²: <b style='color:{bcol};'>"
                f"{best_r2:.4f}</b> — {badge}</div>", unsafe_allow_html=True)

            # Per-model plots
            model_tabs = st.tabs(list(ml_res.keys()))
            for tab_m, (mname, res) in zip(model_tabs, ml_res.items()):
                with tab_m:
                    y_te = np.array(res["y_test"])
                    y_pr = np.array(res["y_pred"])

                    mc1, mc2 = st.columns(2)
                    with mc1:
                        # Actual vs Predicted
                        fig_avp = px.scatter(
                            x=y_te, y=y_pr, opacity=0.4,
                            labels={"x":"Actual Risk Score","y":"Predicted Risk Score"},
                            title=f"{mname}: Actual vs Predicted",
                            color_discrete_sequence=["#2e75b6"])
                        lm = [min(y_te.min(), y_pr.min()), max(y_te.max(), y_pr.max())]
                        fig_avp.add_trace(go.Scatter(x=lm, y=lm, mode="lines",
                            line=dict(color="red", dash="dash"), name="Ideal"))
                        fig_avp.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_avp, use_container_width=True)

                    with mc2:
                        # Residuals
                        resid = y_te - y_pr
                        fig_res = px.scatter(x=y_pr, y=resid, opacity=0.4,
                            labels={"x":"Predicted Risk Score","y":"Residual"},
                            title=f"{mname}: Residuals vs Predicted",
                            color_discrete_sequence=["#c0392b"])
                        fig_res.add_hline(y=0, line_dash="dash", line_color="black")
                        fig_res.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_res, use_container_width=True)

                    # Permutation Feature Importance
                    perm_mean = np.array(res["perm_importance_mean"])
                    perm_std  = np.array(res["perm_importance_std"])
                    order     = np.argsort(perm_mean)
                    fig_imp = go.Figure()
                    fig_imp.add_trace(go.Bar(
                        y=[FEATURE_NAMES[i] for i in order],
                        x=perm_mean[order],
                        orientation="h",
                        error_x=dict(type="data", array=perm_std[order]),
                        marker_color="#8e44ad",
                        name="Permutation Importance",
                    ))
                    fig_imp.update_layout(
                        title=f"{mname}: Permutation Feature Importance (±std)",
                        xaxis_title="Mean Decrease in R² when Feature Permuted",
                        height=360, margin=dict(l=0,r=0,t=40,b=0))
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # SHAP beeswarm summary
                    st.markdown("##### 🔍 SHAP Feature Impact (first 200 test samples)")
                    shap_vals  = np.array(res["shap_values"])
                    shap_X     = np.array(res["shap_X"])
                    shap_means = np.abs(shap_vals).mean(axis=0)
                    shap_order = np.argsort(shap_means)

                    fig_shap = go.Figure()
                    colors_shap = px.colors.diverging.RdBu
                    for idx in shap_order:
                        feat_vals = shap_X[:, idx]
                        sv        = shap_vals[:, idx]
                        norm_fv   = (feat_vals - feat_vals.min()) / max(feat_vals.max() - feat_vals.min(), 1e-9)
                        color_idx = (norm_fv * (len(colors_shap)-1)).astype(int)
                        colors_pt = [colors_shap[c] for c in color_idx]
                        fig_shap.add_trace(go.Scatter(
                            x=sv,
                            y=[FEATURE_NAMES[idx]] * len(sv),
                            mode="markers",
                            marker=dict(color=colors_pt, size=4, opacity=0.5),
                            name=FEATURE_NAMES[idx],
                            showlegend=False,
                            hovertemplate=f"Feature: {FEATURE_NAMES[idx]}<br>SHAP: %{{x:.3f}}<extra></extra>",
                        ))
                    fig_shap.add_vline(x=0, line_dash="dash", line_color="black")
                    fig_shap.update_layout(
                        title=f"{mname}: SHAP Beeswarm (red=high feature value, blue=low)",
                        xaxis_title="SHAP Value (impact on risk prediction)",
                        yaxis_title="Feature",
                        height=400, margin=dict(l=0,r=0,t=50,b=0))
                    st.plotly_chart(fig_shap, use_container_width=True)

            # Cross-model SHAP mean |value| comparison
            st.markdown("#### 🔬 Cross-Model: Mean |SHAP| Feature Ranking")
            shap_compare = []
            for mname, res in ml_res.items():
                sv = np.abs(np.array(res["shap_values"])).mean(axis=0)
                for i, f in enumerate(FEATURE_NAMES):
                    shap_compare.append({"Model": mname, "Feature": f,
                                         "Mean |SHAP|": round(sv[i], 4)})
            shap_cdf = pd.DataFrame(shap_compare)
            fig_shcomp = px.bar(
                shap_cdf.sort_values("Mean |SHAP|", ascending=True),
                x="Mean |SHAP|", y="Feature", color="Model", barmode="group",
                orientation="h",
                color_discrete_map={"Random Forest":"#2e75b6","Gradient Boosting":"#e74c3c"},
                title="Mean |SHAP| Value per Feature — Both Models")
            fig_shcomp.update_layout(height=380, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_shcomp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB: STEP 6 — Scalability Evaluation
# ═══════════════════════════════════════════════════════════════════════════
with tab_step6:
    st.subheader("⚡ Step 6: Scalability & Performance Evaluation")
    st.markdown(
        "<div class='callout-info'>Benchmarks all 4 pipeline stages across increasing "
        "problem sizes. Reports wall-clock time (perf_counter), peak memory (tracemalloc), "
        "and throughput (scenarios/s). Fits a log-log slope to estimate empirical "
        "O(N^k) complexity class.</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout-warn'><b>Research hypothesis:</b> Each pipeline stage should "
        "scale sub-quadratically. Slopes near 1.0 = linear (ideal). "
        "Slopes > 1.5 = bottleneck requiring optimisation.</div>",
        unsafe_allow_html=True)

    if st.button("▶ Run Full Scalability Benchmark", type="primary", key="run_bench"):
        with st.spinner(f"Benchmarking {len(bench_sizes)} problem sizes across 4 pipeline stages…"):
            bench_df = run_scalability_benchmark(
                bench_sizes,
                json.dumps(asset_mix),
                tuple(selected_actors),
                tuple(selected_vecs),
                rng_seed,
                rf_params,
                gb_params,
            )
            st.session_state["bench_results"] = bench_df

    if st.session_state["bench_results"] is None:
        st.info("👆 Click **Run Full Scalability Benchmark** to generate measurements.")
    else:
        bd = st.session_state["bench_results"]

        # KPI row
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Min N",            f"{bd['N'].min():,}")
        k2.metric("Max N",            f"{bd['N'].max():,}")
        k3.metric("Peak Total Time",  f"{bd['total_time'].max():.3f}s")
        k4.metric("Peak Memory",      f"{bd['total_mem'].max():.0f} KB")
        k5.metric("Max Throughput",   f"{bd['throughput'].max():,.0f}/s")

        # 4-panel pipeline dashboard
        st.markdown("#### 📈 Pipeline Scalability Profiles (interactive — hover, zoom)")
        fig_bm = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "⏱ Wall-Clock Time per Stage",
                "📊 Time Breakdown — Stacked Area",
                "🧠 Peak Memory per Stage (KB)",
                "🚀 Throughput (Scenarios / Second)"),
            vertical_spacing=0.14)

        stage_colors = {"gen":"#4285F4","scen":"#34A853","feat":"#FBBC04","ml":"#EA4335"}
        stage_labels = {"gen":"Data Gen","scen":"Scenario Gen",
                        "feat":"Feature Eng.","ml":"ML Train+Infer"}

        for stg, col, lbl in [("gen","#4285F4","Data Gen"),
                               ("scen","#34A853","Scenario Gen"),
                               ("feat","#FBBC04","Feature Eng."),
                               ("ml","#EA4335","ML Train+Infer")]:
            fig_bm.add_trace(go.Scatter(
                x=bd["N"], y=bd[f"{stg}_time"],
                mode="lines+markers", name=lbl,
                line=dict(color=col, width=2)), row=1, col=1)

        # Stacked area
        fig_bm.add_trace(go.Scatter(
            x=bd["N"], y=bd["gen_time"], fill="tozeroy",
            name="Data Gen", line=dict(color="#4285F4"),
            showlegend=False), row=1, col=2)
        for stg, col in [("scen","#34A853"),("feat","#FBBC04"),("ml","#EA4335")]:
            fig_bm.add_trace(go.Scatter(
                x=bd["N"], y=bd[f"{stg}_time"], fill="tonexty",
                name=stage_labels[stg], line=dict(color=col),
                showlegend=False), row=1, col=2)

        for stg, col, sym in [("gen","#4285F4","circle"),
                               ("scen","#34A853","square"),
                               ("feat","#FBBC04","diamond"),
                               ("ml","#EA4335","triangle-up")]:
            fig_bm.add_trace(go.Scatter(
                x=bd["N"], y=bd[f"{stg}_mem"],
                mode="lines+markers", name=stage_labels[stg],
                line=dict(color=col), marker=dict(symbol=sym),
                showlegend=False), row=2, col=1)

        fig_bm.add_trace(go.Bar(
            x=bd["N"], y=bd["throughput"],
            name="Throughput", marker_color="#2980b9",
            showlegend=False), row=2, col=2)

        fig_bm.update_xaxes(title_text="N (Assets)")
        fig_bm.update_yaxes(title_text="Seconds", row=1, col=1)
        fig_bm.update_yaxes(title_text="Seconds", row=1, col=2)
        fig_bm.update_yaxes(title_text="KB",      row=2, col=1)
        fig_bm.update_yaxes(title_text="Scen/s",  row=2, col=2)
        fig_bm.update_layout(height=580, margin=dict(l=0,r=0,t=60,b=0),
                              legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig_bm, use_container_width=True)

        # Per-stage log-log fit
        st.markdown("#### 📐 Empirical O(N^k) Complexity — Log-Log Fit per Stage")
        stage_fits = {}
        fig_ll = go.Figure()
        colors_ll = {"gen":"#4285F4","scen":"#34A853","feat":"#FBBC04","ml":"#EA4335"}
        for stg, lbl in stage_labels.items():
            lN  = np.log(bd["N"].values.astype(float) + 1)
            lT  = np.log(bd[f"{stg}_time"].values.astype(float) + 1e-9)
            slope, intercept = np.polyfit(lN, lT, 1)
            fit  = np.exp(intercept + slope * lN)
            stage_fits[lbl] = slope
            fig_ll.add_trace(go.Scatter(
                x=bd["N"], y=bd[f"{stg}_time"],
                mode="markers", name=f"{lbl} (data)",
                marker=dict(color=colors_ll[stg], size=7)))
            fig_ll.add_trace(go.Scatter(
                x=bd["N"], y=fit, mode="lines",
                name=f"{lbl} fit: O(N^{slope:.2f})",
                line=dict(color=colors_ll[stg], dash="dash", width=1.5)))

        fig_ll.update_layout(
            title="Log-Log Fit: Empirical Complexity per Pipeline Stage",
            xaxis=dict(title="N", type="log"),
            yaxis=dict(title="Time (s)", type="log"),
            height=380, margin=dict(l=0,r=0,t=50,b=0),
            legend=dict(orientation="h", y=-0.22, font=dict(size=9)))
        st.plotly_chart(fig_ll, use_container_width=True)

        # Complexity class table
        st.markdown("#### 🏷️ Complexity Classification by Stage")
        cls_rows = []
        for lbl, slope in stage_fits.items():
            cls_label, cls_col = complexity_class(slope)
            cls_rows.append({
                "Pipeline Stage": lbl,
                "Fitted Slope k": round(slope, 3),
                "Complexity Class": cls_label,
                "Interpretation": (
                    "✅ Scales efficiently — suitable for large deployment"
                    if slope < 1.1 else
                    "⚠️ Minor overhead — monitor at scale"
                    if slope < 1.5 else
                    "🚨 Bottleneck — optimisation required"
                ),
            })
        cls_df = pd.DataFrame(cls_rows)
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

        # Total pipeline complexity
        total_slope = np.polyfit(
            np.log(bd["N"].values.astype(float) + 1),
            np.log(bd["total_time"].values.astype(float) + 1e-9), 1)[0]
        total_class, total_col = complexity_class(total_slope)
        st.markdown(
            f"<div class='callout-info'><b>Overall Pipeline: k = {total_slope:.3f} — "
            f"{total_class}</b><br>"
            f"The full PASTA-ML pipeline (data gen → scenario gen → features → ML) "
            f"scales with empirical complexity O(N^{total_slope:.2f}).</div>",
            unsafe_allow_html=True)

        # Raw benchmark table
        st.markdown("#### 📋 Raw Benchmark Results")
        st.dataframe(bd.style.format({
            "gen_time":   "{:.5f}",
            "scen_time":  "{:.5f}",
            "feat_time":  "{:.5f}",
            "ml_time":    "{:.5f}",
            "total_time": "{:.5f}",
            "gen_mem":    "{:.1f}",
            "scen_mem":   "{:.1f}",
            "feat_mem":   "{:.1f}",
            "ml_mem":     "{:.1f}",
            "total_mem":  "{:.1f}",
            "throughput": "{:,.1f}",
        }), use_container_width=True)

        st.download_button("📥 Download Benchmark Data (CSV)",
                           bd.to_csv(index=False).encode(),
                           "pasta_ml_scalability_benchmark.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
# TAB: EXPORT
# ═══════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("📤 Export All Research Artifacts")
    st.caption("Download all generated data, models metrics, and benchmark results.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📦 Data Artifacts")
        if st.session_state["env"] is not None:
            env = st.session_state["env"]
            st.download_button("📥 Asset Inventory (CSV)",
                env["assets"].to_csv(index=False).encode(),
                "asset_inventory.csv", "text/csv")
            st.download_button("📥 Threat Actor Profiles (CSV)",
                env["actors"].to_csv(index=False).encode(),
                "threat_actors.csv", "text/csv")

        if st.session_state["scenarios"] is not None:
            st.download_button("📥 Threat Scenarios (CSV)",
                st.session_state["scenarios"].to_csv(index=False).encode(),
                "threat_scenarios.csv", "text/csv")

        if st.session_state["features"] is not None:
            st.download_button("📥 Engineered Features + Risk Score (CSV)",
                st.session_state["features"].to_csv(index=False).encode(),
                "engineered_features.csv", "text/csv")

    with col_b:
        st.markdown("#### 📊 Results Artifacts")
        if st.session_state["ml_results"] is not None:
            ml_res = st.session_state["ml_results"]
            summary = []
            for mname, res in ml_res.items():
                summary.append({
                    "Model": mname,
                    "R²": res["r2"], "MAE": res["mae"],
                    "RMSE": res["rmse"], "MAPE(%)": res["mape"],
                    "CV_R2_mean": res["cv_r2_mean"], "CV_R2_std": res["cv_r2_std"],
                    "train_time_s": res["train_time_s"], "infer_ms": res["infer_ms"],
                    "n_train": res["n_train"], "n_test": res["n_test"],
                })
            st.download_button("📥 ML Model Metrics (CSV)",
                pd.DataFrame(summary).to_csv(index=False).encode(),
                "ml_model_metrics.csv", "text/csv")

        if st.session_state["bench_results"] is not None:
            st.download_button("📥 Scalability Benchmark (CSV)",
                st.session_state["bench_results"].to_csv(index=False).encode(),
                "scalability_benchmark.csv", "text/csv")

        # Full config JSON
        config = {
            "n_assets": n_assets, "seed": rng_seed,
            "asset_mix": asset_mix, "threat_actors": selected_actors,
            "n_scenarios": n_scenarios, "attack_vectors": selected_vecs,
            "max_path_len": max_path_len, "test_size": test_size,
            "cv_folds": cv_folds, "rf_params": rf_params, "gb_params": gb_params,
            "bench_sizes": list(bench_sizes),
        }
        st.download_button("🧾 Full Experiment Config (JSON)",
            json.dumps(config, indent=2).encode(), "experiment_config.json", "application/json")

    st.divider()
    st.markdown("#### 📚 Citation & References")
    st.markdown("""
    **Framework:** UcedaVélez, T. & Morana, M.M. (2015). *Risk Centric Threat Modeling*. Wiley.  
    **MITRE ATT&CK:** https://attack.mitre.org  
    **CVSS v3.1:** FIRST.org. https://www.first.org/cvss/v3.1/specification-document  
    **NVD / CVE:** https://nvd.nist.gov  
    **SHAP:** Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.  
    **NetworkX:** Hagberg, A. et al. (2008). *Exploring Network Structure, Dynamics, and Function using NetworkX*.  
    **scikit-learn:** Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825–2830.
    """)
