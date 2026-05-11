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
import io, json, time, tracemalloc, warnings, hashlib, zipfile
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import shap

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
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

# Lightweight ATT&CK/CVE-style enrichment used for defensible synthetic generation.
# These are representative mappings for research simulation, not live threat intel.
ATTACK_VECTOR_ENRICHMENT = {
    "Phishing":              {"mitre_id":"T1566", "requires_credentials":1, "privilege_required":0.1, "epss_mu":0.25},
    "SQLi":                  {"mitre_id":"T1190", "requires_credentials":0, "privilege_required":0.2, "epss_mu":0.45},
    "RCE via Unpatched CVE": {"mitre_id":"T1190", "requires_credentials":0, "privilege_required":0.2, "epss_mu":0.60},
    "Privilege Escalation":  {"mitre_id":"T1068", "requires_credentials":1, "privilege_required":0.5, "epss_mu":0.42},
    "Lateral Movement":      {"mitre_id":"T1021", "requires_credentials":1, "privilege_required":0.5, "epss_mu":0.35},
    "Credential Stuffing":   {"mitre_id":"T1110", "requires_credentials":0, "privilege_required":0.1, "epss_mu":0.30},
    "Supply Chain":          {"mitre_id":"T1195", "requires_credentials":0, "privilege_required":0.4, "epss_mu":0.55},
    "DDoS":                  {"mitre_id":"T1498", "requires_credentials":0, "privilege_required":0.1, "epss_mu":0.20},
    "Data Exfiltration":     {"mitre_id":"T1041", "requires_credentials":1, "privilege_required":0.6, "epss_mu":0.40},
    "Firmware Implant":      {"mitre_id":"T1542", "requires_credentials":1, "privilege_required":0.8, "epss_mu":0.30},
    "Pass-the-Hash":         {"mitre_id":"T1550.002", "requires_credentials":1, "privilege_required":0.5, "epss_mu":0.38},
    "Zero-Day Exploit":      {"mitre_id":"T1203", "requires_credentials":0, "privilege_required":0.2, "epss_mu":0.65},
}

# ── LAYERED TOPOLOGY MAPPING (NEW) ────────────────────────────────────────────
# Maps each asset type to one of three architectural layers (Core / Distribution / Access)
# Core         → backbone, critical (Barabási–Albert scale-free graph)
# Distribution → mid-tier services    (Watts–Strogatz small-world graph)
# Access       → edge / leaf          (random tree)
ASSET_LAYER_MAPPING = {
    "Database Server": "Core",
    "SCADA/ICS":       "Core",
    "Cloud VM":        "Distribution",
    "Enterprise App":  "Distribution",
    "Network Device":  "Distribution",
    "IoT Device":      "Access",
    "Endpoint":        "Access",
}
LAYER_ZONES  = {"Core": "secure",   "Distribution": "internal", "Access": "dmz"}
LAYER_COLORS = {"Core": "#27ae60",  "Distribution": "#2980b9",  "Access": "#c0392b"}
LAYER_ORDINAL = {"Access": 0, "Distribution": 1, "Core": 2}

# Centrality-derived features added to the asset/event records.
CENTRALITY_FEATS = [
    "degree_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    "clustering_coefficient",
]

# Feature set used by the binary attack-vs-normal alerting classifier (Step 5b).
# IMPORTANT: this is INTENTIONALLY disjoint from the formula used to derive the
# regression target risk_score, so the classifier head is not learning the same
# signal as the regression head. The label is derived from whether the asset was
# actually traversed in a Monte-Carlo attack simulation, not from any risk formula.
CLASSIFIER_FEATS = [
    "criticality", "exposure", "patch_compliance", "control_coverage",
    "vuln_count",  "asset_criticality_score",
    "degree_centrality", "betweenness_centrality",
    "eigenvector_centrality", "clustering_coefficient",
    "layer_ord",
]

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
    "env":              None,   # simulated environment (Step 2)
    "scenarios":        None,   # threat scenarios DataFrame (Step 3)
    "features":         None,   # engineered feature DataFrame (Step 4)
    "ml_results":       None,   # trained regression models + metrics (Step 5)
    "bench_results":    None,   # scalability benchmark DataFrame (Step 6)
    "attack_graph":     None,   # legacy attack-graph stats (Step 3)
    # ── NEW: layered topology + Monte-Carlo alerting (Step 5b) ────────────────
    "topology":         None,   # node-link JSON of the layered enterprise graph
    "mc_events":        None,   # event-level dataset (attack + normal)
    "mc_paths":         None,   # list of attack paths from Monte-Carlo runs
    "mc_stats":         None,   # path-length / compromise statistics
    "clf_results":      None,   # trained alerting classifier results
    # ── NEW v3: real-data enrichment + continuous PASTA operations ─────────────
    "real_data_bundle": None,   # uploaded/normalized assets, SBOM, CVE, CTI, controls, labels
    "pif_bundle":       None,   # PASTA Interchange Format export object
    "fair_results":     None,   # FAIR-style financial risk quantification
    "maturity_results": None,   # MM-PASTA maturity assessment
    "freshness_results":None,   # model freshness/drift metrics
    "ticket_backlog":   None,   # DevSecOps-ready remediation tickets
    "review_log":       None,   # human-AI governance review table
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


def safe_div(num, den, default=0.0):
    """Numerically safe division for synthetic/benchmark edge cases."""
    den = float(den) if den is not None else 0.0
    return default if abs(den) < 1e-12 else num / den


def stable_id_int(*parts, modulo=10_000):
    """Stable deterministic ID generator independent of Python hash randomisation."""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16) % modulo


def risk_label_from_score(s):
    if s >= 7.5: return "Critical"
    if s >= 5.0: return "High"
    if s >= 2.5: return "Medium"
    return "Low"


def formula_risk_score(feat):
    """Transparent PASTA baseline risk model. This is a baseline, not ground truth."""
    return (
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


def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mape": round(float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100), 2),
    }


def model_prediction_uncertainty(model, X):
    """Return simple prediction-interval diagnostics for ensemble models.

    For Random Forest, estimator disagreement is used as an epistemic uncertainty
    proxy. For non-ensemble models, NaNs are returned so the UI can still render.
    """
    try:
        estimators = getattr(model, "estimators_", None)
        if not estimators:
            return np.nan, np.nan
        preds = np.vstack([est.predict(X) for est in estimators])
        lower = np.percentile(preds, 5, axis=0)
        upper = np.percentile(preds, 95, axis=0)
        width = upper - lower
        return round(float(np.mean(width)), 4), round(float(np.percentile(width, 90)), 4)
    except Exception:
        return np.nan, np.nan


def target_diagnostics(feat_df):
    """Dataset-level diagnostics to disclose synthetic-target dependency."""
    out = {"target_baseline_corr": np.nan, "target_outcome_corr": np.nan}
    try:
        out["target_baseline_corr"] = round(float(feat_df[["risk_score", "baseline_risk_score"]].corr().iloc[0, 1]), 4)
    except Exception:
        pass
    try:
        out["target_outcome_corr"] = round(float(feat_df[["risk_score", "outcome_risk_score"]].corr().iloc[0, 1]), 4)
    except Exception:
        pass
    return out


def ablation_feature_groups(feat_df, test_size):
    """Small ablation study showing whether each feature family adds value."""
    groups = {
        "Asset only": ["asset_criticality", "exposure_level"],
        "Vulnerability only": ["vuln_count_norm", "cvss_weighted_avg", "exploitability_score", "patch_compliance_inv"],
        "Threat only": ["threat_likelihood", "attacker_capability"],
        "Path/control only": ["attack_path_length_inv", "control_effectiveness_inv"],
        "All features": FEATURE_NAMES,
    }
    rows = []
    y = feat_df["risk_score"].values
    for name, cols in groups.items():
        available = [c for c in cols if c in feat_df.columns]
        if not available:
            continue
        Xg = feat_df[available].fillna(0).values
        Xtr, Xte, ytr, yte = train_test_split(Xg, y, test_size=test_size, random_state=42)
        model = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        rows.append({"Feature Group": name, "Features": ", ".join(available), "R²": round(float(r2_score(yte, model.predict(Xte))), 4)})
    return rows


def mitigation_recommendations(row):
    """Stage-mapped mitigation catalogue for PASTA Stage 7 outputs."""
    recs = []
    if row.get("cvss_weighted_avg", 0) >= 0.75 and row.get("patch_compliance_inv", 0) >= 0.45:
        recs.append(("Patch vulnerable services", "Stage 5", 0.18, "High CVSS combined with weak patch compliance"))
    if row.get("exposure_level", 0) >= 0.70 and row.get("attack_path_length_inv", 0) >= 0.45:
        recs.append(("Reduce exposure / enforce segmentation", "Stage 2/6", 0.20, "Exposed asset has short path toward high-value targets"))
    if row.get("control_effectiveness_inv", 0) >= 0.45:
        recs.append(("Increase compensating control coverage", "Stage 7", 0.14, "Low control coverage increases residual risk"))
    if row.get("threat_likelihood", 0) >= 0.55 or row.get("attacker_capability", 0) >= 0.75:
        recs.append(("Add monitoring and threat-hunting controls", "Stage 4/7", 0.12, "Likely/capable actor profile requires detection response"))
    if row.get("vuln_count_norm", 0) >= 0.65:
        recs.append(("Prioritise vulnerability backlog reduction", "Stage 5", 0.10, "Large normalized vulnerability population"))
    if not recs:
        recs.append(("Maintain baseline controls and continuous validation", "Stage 7", 0.04, "No single dominant risk driver detected"))
    recs = sorted(recs, key=lambda x: x[2], reverse=True)[:3]
    actions = "; ".join(r[0] for r in recs)
    stages = "; ".join(sorted({r[1] for r in recs}))
    rationale = "; ".join(r[3] for r in recs)
    reduction = min(0.45, sum(r[2] for r in recs))
    return actions, stages, rationale, reduction

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# LAYERED TOPOLOGY + CENTRALITY ENGINE  (NEW — Home.py ideas #1 + #2)
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
def _random_tree(n, seed):
    """Version-agnostic random tree (avoids networkx API drift across versions)."""
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    if n <= 0:
        return G
    G.add_node(0)
    for i in range(1, n):
        parent = int(rng.integers(0, i))
        G.add_edge(parent, i)
    return G


@st.cache_data(show_spinner=False)
def build_layered_topology(ids_by_layer_json, seed):
    """
    Build the enterprise graph as three composed sub-graphs, one per layer:
      • Core         → Barabási–Albert (scale-free, hub-and-spoke)
      • Distribution → Watts–Strogatz   (small-world)
      • Access       → Random tree      (edge / leaf)
    Then deterministically inter-link layers (Access → Distribution → Core).

    Returns: node-link JSON of an nx.DiGraph where every node carries
    `asset_id`, `layer`, `zone` attributes. Edge direction encodes attack flow
    (inward, from Access toward Core).
    """
    layers = json.loads(ids_by_layer_json)
    core_ids   = layers.get("Core", [])
    dist_ids   = layers.get("Distribution", [])
    access_ids = layers.get("Access", [])

    # ── per-layer graphs ──────────────────────────────────────────────────────
    # Core: scale-free hubs
    if len(core_ids) >= 3:
        Gc_int = nx.barabasi_albert_graph(len(core_ids),
                                          min(2, len(core_ids) - 1),
                                          seed=int(seed))
    else:
        Gc_int = nx.path_graph(max(1, len(core_ids)))
    Gc = nx.relabel_nodes(Gc_int, dict(enumerate(core_ids)))

    # Distribution: small-world
    k_ws = min(4, max(2, len(dist_ids) - 1))
    if len(dist_ids) >= 4:
        Gd_int = nx.watts_strogatz_graph(len(dist_ids), k_ws, 0.2,
                                         seed=int(seed))
    else:
        Gd_int = nx.path_graph(max(1, len(dist_ids)))
    Gd = nx.relabel_nodes(Gd_int, dict(enumerate(dist_ids)))

    # Access: random tree
    Ga_int = _random_tree(len(access_ids), seed=int(seed))
    Ga = nx.relabel_nodes(Ga_int, dict(enumerate(access_ids)))

    G_und = nx.compose_all([Gc, Gd, Ga]) if (core_ids or dist_ids or access_ids) else nx.Graph()

    # Annotate layer / zone
    for n in core_ids:   G_und.add_node(n); G_und.nodes[n]["layer"] = "Core";         G_und.nodes[n]["zone"] = LAYER_ZONES["Core"]
    for n in dist_ids:   G_und.add_node(n); G_und.nodes[n]["layer"] = "Distribution"; G_und.nodes[n]["zone"] = LAYER_ZONES["Distribution"]
    for n in access_ids: G_und.add_node(n); G_und.nodes[n]["layer"] = "Access";       G_und.nodes[n]["zone"] = LAYER_ZONES["Access"]

    # Inter-layer wiring: Distribution → Core, Access → Distribution
    for i, n in enumerate(dist_ids):
        if core_ids:   G_und.add_edge(n, core_ids[i % len(core_ids)])
    for i, n in enumerate(access_ids):
        if dist_ids:   G_und.add_edge(n, dist_ids[i % len(dist_ids)])

    # Directed: attack-flow direction is inward (lower layer → higher layer)
    G = nx.DiGraph()
    for n, data in G_und.nodes(data=True):
        G.add_node(n, **data)
    for u, v in G_und.edges():
        lu = G_und.nodes[u].get("layer", "Distribution")
        lv = G_und.nodes[v].get("layer", "Distribution")
        ru, rv = LAYER_ORDINAL.get(lu, 1), LAYER_ORDINAL.get(lv, 1)
        if ru < rv:        G.add_edge(u, v)            # outer → inner
        elif ru > rv:      G.add_edge(v, u)
        else:              G.add_edge(u, v); G.add_edge(v, u)  # peer (intra-layer)
    return json.dumps(nx.node_link_data(G))


@st.cache_data(show_spinner=False)
def compute_centrality_features(topology_json):
    """
    Derive graph-structural features per node from the layered topology.
    These complement (do not replace) the existing asset-intrinsic features.
    Uses eigenvector_centrality_numpy with a PageRank fallback for robustness.
    """
    G = nx.node_link_graph(json.loads(topology_json))
    G_und = G.to_undirected()

    dc = nx.degree_centrality(G)
    bc = nx.betweenness_centrality(G, normalized=True)
    try:
        ec = nx.eigenvector_centrality_numpy(G)
    except Exception:
        ec = nx.pagerank(G)
    cc = nx.clustering(G_und)

    rows = []
    for n in G.nodes:
        rows.append({
            "asset_id":               n,
            "degree_centrality":      round(float(dc.get(n, 0.0)), 6),
            "betweenness_centrality": round(float(bc.get(n, 0.0)), 6),
            "eigenvector_centrality": round(float(ec.get(n, 0.0)), 6),
            "clustering_coefficient": round(float(cc.get(n, 0.0)), 6),
            "layer":                  G.nodes[n].get("layer", "Distribution"),
            "zone":                   G.nodes[n].get("zone",  "internal"),
        })
    return pd.DataFrame(rows)


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

    # ── NEW: layer / zone assignment + centrality features ──────────────────
    # Each asset is mapped to one of three architectural layers (Core / Dist / Access)
    # based on its type, then the BA + WS + Tree composite topology is built and
    # graph-structural centrality features are merged into the asset record.
    asset_df["layer"] = asset_df["asset_type"].map(ASSET_LAYER_MAPPING).fillna("Distribution")
    asset_df["zone"]  = asset_df["layer"].map(LAYER_ZONES).fillna("internal")
    asset_df["layer_ord"] = asset_df["layer"].map(LAYER_ORDINAL).fillna(1).astype(int)

    ids_by_layer = {
        layer: asset_df.loc[asset_df["layer"] == layer, "asset_id"].tolist()
        for layer in ["Core", "Distribution", "Access"]
    }
    topology_json = build_layered_topology(json.dumps(ids_by_layer), int(seed))
    cent_df = compute_centrality_features(topology_json)
    asset_df = asset_df.merge(
        cent_df[["asset_id"] + CENTRALITY_FEATS], on="asset_id", how="left"
    )
    for f in CENTRALITY_FEATS:
        asset_df[f] = asset_df[f].fillna(0.0)

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
            "n_assets": len(asset_df), "n_actors": len(actor_df),
            "topology_json": topology_json}

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 ENGINE — Threat Scenario Generation + Attack Graph
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_scenarios(env_assets_json, env_actors_json, topology_json,
                       n_scenarios, selected_vectors, seed, max_path_len):
    """Generate synthetic threat scenarios on the layered enterprise topology.

    Uses the BA + WS + Tree composite graph built in Step 2 as the attack graph
    (replacing the prior random Poisson-edge graph). Each edge is annotated
    with an attack vector + CVSS-derived weight for Dijkstra shortest-path
    computation.
    """
    rng  = np.random.default_rng(seed)
    asset_df = pd.read_json(io.StringIO(env_assets_json), orient="records")
    actor_df = pd.read_json(io.StringIO(env_actors_json), orient="records")

    n_assets = len(asset_df)

    # ── Attack graph: the layered topology, annotated with attack-vector edges ──
    G = nx.node_link_graph(json.loads(topology_json))
    for u, v in list(G.edges()):
        vec    = rng.choice(selected_vectors)
        diff   = ATTACK_VECTORS[vec]["difficulty"]
        cvss_e = ATTACK_VECTORS[vec]["cvss_base"]
        G[u][v]["vector"]     = vec
        G[u][v]["difficulty"] = float(diff)
        G[u][v]["weight"]     = float(1.0 - (cvss_e / 10.0))
        G[u][v]["cvss"]       = float(cvss_e)

    # Sample attack paths for scenarios — entry points are exposed assets,
    # high-value targets are the top-criticality nodes (typically in Core).
    entry_points = asset_df[asset_df["exposure"] > 0.6]["asset_id"].tolist()
    if not entry_points:
        entry_points = asset_df["asset_id"].head(min(5, n_assets)).tolist()
    high_value = (
        asset_df.nlargest(max(3, n_assets // 5), "asset_criticality_score")["asset_id"].tolist()
    )

    rows = []
    for _ in range(n_scenarios):
        actor_row  = actor_df.sample(1, random_state=int(rng.integers(0,9999))).iloc[0]
        asset_row  = asset_df.sample(1, random_state=int(rng.integers(0,9999))).iloc[0]
        vec        = rng.choice(selected_vectors)
        vec_cfg    = ATTACK_VECTORS[vec]
        enrich    = ATTACK_VECTOR_ENRICHMENT.get(vec, {})

        # CVSS score — NVD-calibrated mixture model
        sev_roll = rng.random()
        if   sev_roll < 0.14: cvss = float(rng.uniform(9.0, 10.0))  # Critical
        elif sev_roll < 0.48: cvss = float(rng.uniform(7.0,  9.0))  # High
        elif sev_roll < 0.98: cvss = float(rng.uniform(4.0,  7.0))  # Medium
        else:                  cvss = float(rng.uniform(0.1,  4.0))  # Low

        exploitability  = float(rng.beta(4, 3))     # 0–1, skewed high
        attack_complexity = float(rng.uniform(0.2, 1.0))
        epss_probability = float(np.clip(rng.normal(enrich.get("epss_mu", 0.35), 0.12), 0.01, 0.95))
        privilege_required = float(enrich.get("privilege_required", 0.3))
        requires_credentials = int(enrich.get("requires_credentials", 0))
        network_reachability = float(np.clip(asset_row["exposure"] * (1 - attack_complexity * 0.25), 0, 1))
        segmentation_control = float(np.clip(asset_row["control_coverage"] * (1.0 if asset_row.get("layer", "") == "Core" else 0.75), 0, 1))
        exploit_maturity = str(rng.choice(["Proof-of-Concept", "Functional", "High", "Weaponized"], p=[0.25, 0.35, 0.25, 0.15]))
        cve_id = f"CVE-202{int(rng.integers(1, 6))}-{stable_id_int(vec, asset_row['asset_type'], seed, _, modulo=90000)+10000}"
        impact_score    = float((cvss / 10.0) * asset_row["asset_criticality_score"])

        # Shortest path length in attack graph (asset_id-keyed)
        src = str(rng.choice(entry_points))
        tgt = str(rng.choice(high_value))
        try:
            path_len = nx.shortest_path_length(G, source=src, target=tgt, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_len = float(rng.integers(2, max_path_len + 1))
        path_len = max(1.0, path_len)

        # Threat likelihood
        threat_likelihood = float(
            actor_row["capability"] * exploitability * epss_probability *
            network_reachability * (1 - segmentation_control * 0.35) *
            (1 - attack_complexity * 0.2)
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
            "mitre_technique":     enrich.get("mitre_id", "T0000"),
            "attack_tactic":       vec_cfg.get("tactic", "Unknown"),
            "representative_cve":  cve_id,
            "epss_probability":    round(epss_probability, 3),
            "exploit_maturity":    exploit_maturity,
            "requires_credentials":requires_credentials,
            "privilege_required":  round(privilege_required, 3),
            "network_reachability":round(network_reachability, 3),
            "segmentation_control":round(segmentation_control, 3),
            "source_asset":        src,
            "target_asset":        tgt,
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
    """Extract engineered features and create defensible baseline/outcome labels.

    Important research design change:
      • baseline_risk_score = transparent PASTA weighted formula.
      • risk_score = hybrid target that blends baseline + simulation/outcome drivers.
      • mitigation_* columns provide Stage-7 actionable outputs.

    This avoids presenting the formula itself as the only ground truth.
    """
    df = pd.read_json(io.StringIO(scenarios_json), orient="records")

    feat = pd.DataFrame()
    max_vuln = max(float(df["vuln_count"].max()), 1.0)

    feat["asset_criticality"]        = df["asset_criticality"].clip(0, 1)
    feat["vuln_count_norm"]          = np.log1p(df["vuln_count"]) / max(np.log1p(max_vuln), 1e-9)
    feat["cvss_weighted_avg"]        = df["cvss_score"].clip(0, 10) / 10.0
    feat["exploitability_score"]     = df["exploitability"].clip(0, 1)

    path_inv = 1.0 / df["attack_path_length"].clip(lower=0.5)
    feat["attack_path_length_inv"]   = path_inv / max(float(path_inv.max()), 1e-9)
    feat["threat_likelihood"]        = df["threat_likelihood"].clip(0, 1)
    feat["exposure_level"]           = df["exposure"].clip(0, 1)
    feat["patch_compliance_inv"]     = 1.0 - df["patch_compliance"].clip(0, 1)
    feat["attacker_capability"]      = df["actor_capability"].clip(0, 1)
    feat["control_effectiveness_inv"]= 1.0 - df["control_coverage"].clip(0, 1)

    # Additional scenario/outcome fields carried for validation and analysis.
    feat["epss_probability"]         = df.get("epss_probability", 0.35).clip(0, 1)
    feat["network_reachability"]     = df.get("network_reachability", df["exposure"]).clip(0, 1)
    feat["segmentation_control"]     = df.get("segmentation_control", df["control_coverage"]).clip(0, 1)
    feat["privilege_required"]       = df.get("privilege_required", 0.3).clip(0, 1)
    feat["requires_credentials"]     = df.get("requires_credentials", 0).astype(int)
    # Impact proxy is intentionally kept outside FEATURE_NAMES. It contributes to
    # target realism, but is not directly given to the regression model.
    feat["impact_proxy"]             = (df.get("impact_score", feat["asset_criticality"]).clip(0, 1) * 10).round(3)

    # Transparent formula baseline retained for reviewer transparency.
    feat["baseline_risk_score"] = np.clip(formula_risk_score(feat), 0, 10).round(3)

    # Simulation/outcome-inspired risk: includes hidden/semi-observed drivers, so ML is
    # not merely re-learning the baseline formula.
    maturity_weight = df.get("exploit_maturity", pd.Series(["Functional"] * len(df))).map({
        "Proof-of-Concept": 0.55, "Functional": 0.70, "High": 0.85, "Weaponized": 1.00
    }).fillna(0.70).astype(float)

    outcome_prob = (
        0.22 * feat["epss_probability"] +
        0.18 * feat["network_reachability"] +
        0.16 * feat["exploitability_score"] +
        0.14 * feat["attacker_capability"] +
        0.12 * feat["asset_criticality"] +
        0.10 * feat["patch_compliance_inv"] +
        0.08 * (1 - feat["segmentation_control"])
    ) * maturity_weight * (1 - 0.15 * feat["privilege_required"])
    feat["outcome_risk_score"] = np.clip(outcome_prob * 10, 0, 10).round(3)

    # Hybrid target: baseline + outcome + deterministic heterogeneity noise.
    noise_rng = np.random.default_rng(42)
    heterogeneity = noise_rng.normal(0, 0.28, len(feat))
    feat["risk_score"] = np.clip(
        0.55 * feat["baseline_risk_score"] +
        0.35 * feat["outcome_risk_score"] +
        0.10 * feat["impact_proxy"] +
        heterogeneity,
        0, 10
    ).round(3)

    feat["risk_label"] = feat["risk_score"].apply(risk_label_from_score)
    feat["baseline_risk_label"] = feat["baseline_risk_score"].apply(risk_label_from_score)

    # Stage-7 mitigation catalogue and residual risk estimate.
    mitig = feat.apply(mitigation_recommendations, axis=1, result_type="expand")
    mitig.columns = ["mitigation_actions", "mitigation_stages", "mitigation_rationale", "risk_reduction_factor"]
    feat = pd.concat([feat, mitig], axis=1)
    feat["residual_risk_score"] = np.clip(feat["risk_score"] * (1 - feat["risk_reduction_factor"]), 0, 10).round(3)

    # Carry over categorical/research traceability columns for EDA and grouped validation.
    for col in [
        "actor_type", "asset_type", "attack_vector", "cvss_severity",
        "mitre_technique", "attack_tactic", "representative_cve",
        "exploit_maturity", "source_asset", "target_asset"
    ]:
        if col in df.columns:
            feat[col] = df[col].values

    return feat

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 ENGINE — ML Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def train_models(features_json, rf_params, gb_params, test_size, cv_folds):
    """Train baseline and ML regressors with stronger validation diagnostics."""
    feat_df = pd.read_json(io.StringIO(features_json), orient="records")

    X = feat_df[FEATURE_NAMES].fillna(0).values
    y = feat_df["risk_score"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(feat_df)), test_size=test_size, random_state=42
    )

    results = {}
    diagnostics = target_diagnostics(feat_df)
    ablation_rows = ablation_feature_groups(feat_df, test_size)

    # Reviewer-friendly baselines first.
    baseline_pred = feat_df.iloc[idx_test].get("baseline_risk_score", pd.Series(np.repeat(np.mean(y_train), len(idx_test)))).values
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    baseline_specs = [
        ("Mean Dummy Baseline", dummy.predict(X_test), np.zeros(len(FEATURE_NAMES)).tolist()),
        ("PASTA Formula Baseline", baseline_pred, [0.20,0.15,0.15,0.12,0.10,0.10,0.08,0.05,0.03,0.02]),
    ]
    for name, pred, imp in baseline_specs:
        metrics = evaluate_regression(y_test, pred)
        results[name] = {
            "model_kind": "baseline",
            "target_baseline_corr": diagnostics.get("target_baseline_corr", np.nan),
            "target_outcome_corr": diagnostics.get("target_outcome_corr", np.nan),
            "ablation_rows": ablation_rows,
            "uncertainty_mean_width": np.nan,
            "uncertainty_p90_width": np.nan,
            "y_test": y_test.tolist(),
            "y_pred": np.asarray(pred).tolist(),
            **metrics,
            "cv_r2_mean": np.nan, "cv_r2_std": np.nan,
            "group_asset_r2": np.nan, "group_vector_r2": np.nan,
            "train_time_s": 0.0, "infer_ms": 0.0,
            "perm_importance_mean": imp,
            "perm_importance_std": [0.0] * len(FEATURE_NAMES),
            "shap_values": np.zeros((min(200, len(X_test)), len(FEATURE_NAMES))).tolist(),
            "shap_X": X_test[:200].tolist(),
            "n_train": len(X_train), "n_test": len(X_test),
        }

    candidates = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(**gb_params, random_state=42)),
    ]

    def grouped_holdout_score(model, group_col):
        if group_col not in feat_df.columns or feat_df[group_col].nunique() < 2:
            return np.nan
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            groups = feat_df[group_col].fillna("Unknown").astype(str).values
            tr, te = next(splitter.split(X, y, groups))
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            return round(float(r2_score(y[te], pred)), 4)
        except Exception:
            return np.nan

    for name, model in candidates:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(X_test)
        infer_time = (time.perf_counter() - t1) * 1000

        metrics = evaluate_regression(y_test, y_pred)

        try:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="r2", n_jobs=-1)
            cv_mean = round(float(cv_scores.mean()), 4)
            cv_std = round(float(cv_scores.std()), 4)
        except Exception:
            cv_mean = np.nan; cv_std = np.nan

        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
            perm_mean = perm.importances_mean.tolist()
            perm_std = perm.importances_std.tolist()
        except Exception:
            perm_mean = np.zeros(len(FEATURE_NAMES)).tolist()
            perm_std = np.zeros(len(FEATURE_NAMES)).tolist()

        # SHAP for tree models; fallback to coefficient-style approximation for linear models.
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test[:200])
        except Exception:
            coef = getattr(model, "coef_", np.zeros(len(FEATURE_NAMES)))
            shap_vals = (X_test[:200] - X_train.mean(axis=0)) * coef

        # Use fresh model instances for grouped holdout to avoid mutating fitted model.
        if name == "Linear Regression":
            gh_model_1 = LinearRegression(); gh_model_2 = LinearRegression()
        elif name == "Random Forest":
            gh_model_1 = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
            gh_model_2 = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
        else:
            gh_model_1 = GradientBoostingRegressor(**gb_params, random_state=42)
            gh_model_2 = GradientBoostingRegressor(**gb_params, random_state=42)

        uncertainty_mean_width, uncertainty_p90_width = model_prediction_uncertainty(model, X_test)

        results[name] = {
            "model_kind": "ml",
            "target_baseline_corr": diagnostics.get("target_baseline_corr", np.nan),
            "target_outcome_corr": diagnostics.get("target_outcome_corr", np.nan),
            "ablation_rows": ablation_rows,
            "uncertainty_mean_width": uncertainty_mean_width,
            "uncertainty_p90_width": uncertainty_p90_width,
            "model": model,
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            **metrics,
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "group_asset_r2": grouped_holdout_score(gh_model_1, "asset_type"),
            "group_vector_r2": grouped_holdout_score(gh_model_2, "attack_vector"),
            "group_mitre_r2": grouped_holdout_score(gh_model_2, "mitre_technique"),
            "train_time_s": round(train_time, 4),
            "infer_ms": round(infer_time, 3),
            "perm_importance_mean": perm_mean,
            "perm_importance_std": perm_std,
            "shap_values": np.asarray(shap_vals).tolist(),
            "shap_X": X_test[:200].tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    return results

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# STEP 5b ENGINE — Monte-Carlo Attack Simulation + Alerting Classifier
#                  (NEW — Home.py ideas #3 + #4)
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def monte_carlo_attack_simulation(env_assets_json, topology_json,
                                   n_simulations, attack_steps,
                                   epsilon, seed, normal_alert_rate=0.05):
    """Run K independent ε-greedy attacker simulations on the layered topology.

    Produces:
      • An event-level dataset with one row per (sim, step, asset) for both
        attack events (assets traversed by the attacker) and normal events
        (random benign asset accesses, matched 1:1 by count per simulation).
      • A path-statistics dict (mean / std / p95 length, core compromise rate).
      • The list of attack paths.

    Class label design — IMPORTANT for thesis defence:
      `alert = 1` for attack events is derived from whether the asset was
      actually traversed by the attacker, NOT from the regression risk_score.
      This avoids the target-leakage issue present in many synthetic security
      datasets where both heads of a dual-task model end up learning the same
      formula.
    """
    rng = np.random.default_rng(seed)
    asset_df = pd.read_json(io.StringIO(env_assets_json), orient="records")
    G = nx.node_link_graph(json.loads(topology_json))

    # Indexable lookup for fast per-step feature emission
    asset_lookup = asset_df.set_index("asset_id").to_dict("index")
    max_vuln     = max(int(asset_df["vuln_count"].max()), 1)

    def node_score(node_id):
        """ε-greedy attacker's attractiveness function over neighbour candidates."""
        a = asset_lookup.get(node_id, {})
        return (
            0.30 * a.get("vuln_count", 0) / max_vuln +
            0.25 * (1.0 - a.get("patch_compliance", 0.5)) +
            0.20 * a.get("exposure", 0.5) +
            0.15 * a.get("criticality", 0.5) +
            0.10 * a.get("betweenness_centrality", 0.0)
        )

    # Entry pool: highest-exposure assets (typically Access layer)
    entry_pool = (
        asset_df.nlargest(max(3, len(asset_df) // 10), "exposure")["asset_id"].tolist()
    )
    if not entry_pool:
        entry_pool = asset_df["asset_id"].head(min(5, len(asset_df))).tolist()

    attack_paths = []
    events       = []
    base_time    = datetime(2026, 1, 1, 0, 0, 0)

    NODE_FIELDS = [
        "asset_type", "layer", "zone",
        "criticality", "exposure", "patch_compliance", "control_coverage",
        "vuln_count",  "asset_criticality_score",
        "degree_centrality", "betweenness_centrality",
        "eigenvector_centrality", "clustering_coefficient",
        "layer_ord",
    ]

    def emit(sim_id, step_idx, asset_id, label, alert_val, ts_offset):
        a = asset_lookup.get(asset_id, {})
        row = {"simulation": sim_id, "step": step_idx, "asset_id": asset_id,
               "label": label, "alert": int(alert_val),
               "timestamp": (base_time + timedelta(seconds=ts_offset)).isoformat()}
        for fld in NODE_FIELDS:
            row[fld] = a.get(fld, 0 if fld not in ("asset_type","layer","zone") else "Unknown")
        return row

    for sim in range(n_simulations):
        start = str(rng.choice(entry_pool))
        path = [start]
        visited = {start}

        for step in range(attack_steps):
            # Successors first (attack-flow direction); fall back to predecessors if stuck
            cands = [n for n in G.successors(path[-1]) if n not in visited]
            if not cands:
                cands = [n for n in G.predecessors(path[-1]) if n not in visited]
            if not cands:
                break
            if rng.random() < epsilon:
                nxt = str(rng.choice(cands))
            else:
                nxt = max(cands, key=node_score)
            path.append(nxt)
            visited.add(nxt)

        attack_paths.append(path)

        # Attack events
        for s_idx, aid in enumerate(path):
            events.append(emit(sim, s_idx, aid, "attack", 1,
                               sim * 1000 + s_idx * 5))

        # Matched normal events (count = len(path); low false-alarm rate on normals)
        n_normal = len(path)
        normal_ids = rng.choice(asset_df["asset_id"].tolist(),
                                size=min(n_normal, len(asset_df)),
                                replace=False)
        for nid in normal_ids:
            alert_val = 1 if rng.random() < normal_alert_rate else 0
            events.append(emit(sim, 0, str(nid), "normal", alert_val,
                               sim * 1000 + 500))

    events_df = pd.DataFrame(events)

    path_lengths = [len(p) for p in attack_paths]
    compromised  = set().union(*[set(p) for p in attack_paths]) if attack_paths else set()
    core_breaches = [
        any(asset_lookup.get(n, {}).get("layer") == "Core" for n in p)
        for p in attack_paths
    ]

    stats = {
        "n_simulations":            len(attack_paths),
        "mean_path_length":         float(np.mean(path_lengths))   if path_lengths else 0.0,
        "std_path_length":          float(np.std(path_lengths))    if path_lengths else 0.0,
        "p95_path_length":          float(np.percentile(path_lengths, 95)) if path_lengths else 0.0,
        "max_path_length":          int(np.max(path_lengths))      if path_lengths else 0,
        "unique_assets_compromised":int(len(compromised)),
        "core_compromise_rate":     float(np.mean(core_breaches))  if core_breaches else 0.0,
        "epsilon":                  float(epsilon),
        "attack_event_count":       int(events_df["label"].eq("attack").sum()),
        "normal_event_count":       int(events_df["label"].eq("normal").sum()),
    }

    return events_df, attack_paths, stats


@st.cache_data(show_spinner=False)
def train_alert_classifier(events_json, test_size, cv_folds):
    """Train RF + GB classifiers on the attack-vs-normal event-level dataset.

    Reports operationally-relevant metrics (precision / recall / F1 / ROC-AUC /
    PR-AUC) and exposes confusion matrices and predicted probabilities for
    threshold analysis. Uses `class_weight='balanced'` on the RF and stratified
    splitting to handle the natural class imbalance.
    """
    df = pd.read_json(io.StringIO(events_json), orient="records")
    df["layer_ord"] = df.get("layer_ord", df["layer"].map(LAYER_ORDINAL).fillna(1)).astype(int)

    feats = [f for f in CLASSIFIER_FEATS if f in df.columns]
    X = df[feats].fillna(0).values
    y = (df["label"] == "attack").astype(int).values

    if len(np.unique(y)) < 2:
        return {"error": "Only one class present — cannot train a classifier."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    results = {}
    candidates = [
        ("RF Classifier",
         RandomForestClassifier(n_estimators=150, max_depth=None,
                                class_weight="balanced",
                                random_state=42, n_jobs=-1)),
        ("GB Classifier",
         GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                    learning_rate=0.1,
                                    random_state=42)),
    ]
    for name, model in candidates:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred  = model.predict(X_test)
        infer_ms = (time.perf_counter() - t1) * 1000

        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = y_pred.astype(float)

        # Stratified k-fold CV F1
        try:
            cv_f1 = cross_val_score(model, X, y, cv=cv_folds,
                                    scoring="f1", n_jobs=-1)
            cv_f1_mean, cv_f1_std = float(cv_f1.mean()), float(cv_f1.std())
        except Exception:
            cv_f1_mean, cv_f1_std = float("nan"), float("nan")

        results[name] = {
            "accuracy":     round(accuracy_score(y_test, y_pred), 4),
            "precision":    round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":       round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":           round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":      round(roc_auc_score(y_test, y_proba), 4),
            "pr_auc":       round(average_precision_score(y_test, y_proba), 4),
            "cv_f1_mean":   round(cv_f1_mean, 4),
            "cv_f1_std":    round(cv_f1_std,  4),
            "confusion":    confusion_matrix(y_test, y_pred).tolist(),
            "y_test":       y_test.tolist(),
            "y_pred":       y_pred.tolist(),
            "y_proba":      y_proba.tolist(),
            "feat_imp":     model.feature_importances_.tolist(),
            "feat_names":   feats,
            "train_time_s": round(train_time, 4),
            "infer_ms":     round(infer_ms, 3),
            "n_train":      int(len(X_train)),
            "n_test":       int(len(X_test)),
            "class_balance":{"attack": int(np.sum(y_train==1)),
                             "normal": int(np.sum(y_train==0))},
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
        topology_json = env["topology_json"]
        tracemalloc.start(); t0 = time.perf_counter()
        sc_df, _ = generate_scenarios(assets_json, actors_json, topology_json,
                                       n_sc, base_vectors, seed, max_path_len=8)
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
# v3 EXTENSIONS — REAL DATA, PIF, CTI, FAIR, MM-PASTA, DRIFT, TICKETS
# These functions are intentionally defensive: if uploaded real-data files are
# missing, the app keeps using the synthetic/simulation pipeline without errors.
# ═══════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

PIF_VERSION = "PASTA-IF/0.4"


# Official/public data sources that can be referenced from the app. These are
# not fetched automatically by default to keep the app reproducible/offline-safe;
# the links and schemas make it easy to download/source real data on demand.
REAL_DATA_SOURCE_CATALOG = [
    {
        "Source": "NVD CVE API / Feeds",
        "PASTA Stage": "V - Vulnerability Analysis",
        "Use in App": "Populate vulnerabilities.csv with cve_id, cvss_score and affected components/assets.",
        "Official URL": "https://nvd.nist.gov/developers/vulnerabilities",
        "Suggested File": "vulnerabilities.csv",
        "Key Fields": "cve_id, cvss_score, published_date, last_modified, cwe, affected_product",
    },
    {
        "Source": "CISA Known Exploited Vulnerabilities (KEV)",
        "PASTA Stage": "V/VII - Vulnerability + Risk Prioritization",
        "Use in App": "Set known_exploited=1 for CVEs in the KEV catalog.",
        "Official URL": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
        "Suggested File": "vulnerabilities.csv",
        "Key Fields": "cve_id, vendor_project, product, date_added, due_date, known_exploited",
    },
    {
        "Source": "FIRST EPSS",
        "PASTA Stage": "V/VII - Exploit Likelihood",
        "Use in App": "Populate epss_score to estimate real-world exploitation probability.",
        "Official URL": "https://www.first.org/epss/",
        "Suggested File": "vulnerabilities.csv",
        "Key Fields": "cve_id, epss_score, percentile",
    },
    {
        "Source": "MITRE ATT&CK Enterprise Matrix",
        "PASTA Stage": "IV/VI - Threat Analysis + Attack Simulation",
        "Use in App": "Populate cti.csv / mitre_mapping.csv with tactic and technique IDs.",
        "Official URL": "https://attack.mitre.org/",
        "Suggested File": "cti.csv",
        "Key Fields": "mitre_technique, tactic, threat_actor, target_asset_type, confidence",
    },
    {
        "Source": "OASIS STIX/TAXII Standards",
        "PASTA Stage": "IV - CTI Ingestion",
        "Use in App": "Use as the schema reference for CTI import/export design.",
        "Official URL": "https://oasis-open.github.io/cti-documentation/",
        "Suggested File": "cti.csv or stix_bundle.json",
        "Key Fields": "indicator, relationship, threat_actor, attack_pattern, observed_data",
    },
    {
        "Source": "CycloneDX SBOM Standard",
        "PASTA Stage": "II/V - Scope + CVE-to-Component Mapping",
        "Use in App": "Populate sbom.csv from CycloneDX SBOM exports.",
        "Official URL": "https://cyclonedx.org/specification/overview/",
        "Suggested File": "sbom.csv",
        "Key Fields": "asset_id, component_name, component_version, package_type, cve_id",
    },
    {
        "Source": "SPDX SBOM Standard",
        "PASTA Stage": "II/V - Scope + Supply Chain",
        "Use in App": "Alternative SBOM format reference for component inventories.",
        "Official URL": "https://spdx.dev/",
        "Suggested File": "sbom.csv",
        "Key Fields": "asset_id, component_name, component_version, package_type, license, supplier",
    },
    {
        "Source": "NIST NVD CVSS Specification Reference",
        "PASTA Stage": "V/VII - Severity + Risk Scoring",
        "Use in App": "Use CVSS base score/vector fields in vulnerability enrichment.",
        "Official URL": "https://www.first.org/cvss/",
        "Suggested File": "vulnerabilities.csv",
        "Key Fields": "cve_id, cvss_score, cvss_vector, attack_complexity, privileges_required",
    },
]

CSV_TEMPLATE_ROWS = {
    "assets.csv": [
        {"asset_id":"WEB-001","asset_type":"Web Server","zone":"DMZ","criticality":0.85,"exposure":0.95,"patch_compliance":0.55,"control_coverage":0.60,"asset_value":250000,"data_source":"CMDB / cloud inventory","source_reference":"internal CMDB export"},
        {"asset_id":"DB-001","asset_type":"Database Server","zone":"Core","criticality":0.95,"exposure":0.30,"patch_compliance":0.70,"control_coverage":0.80,"asset_value":750000,"data_source":"CMDB / cloud inventory","source_reference":"internal CMDB export"},
    ],
    "sbom.csv": [
        {"asset_id":"WEB-001","component_name":"Apache HTTP Server","component_version":"2.4.x","package_type":"application","cve_id":"CVE-2021-41773","data_source":"CycloneDX/SPDX SBOM","source_reference":"SBOM export"},
        {"asset_id":"WEB-001","component_name":"OpenSSL","component_version":"3.0.x","package_type":"library","cve_id":"CVE-2022-3602","data_source":"CycloneDX/SPDX SBOM","source_reference":"SBOM export"},
    ],
    "vulnerabilities.csv": [
        {"asset_id":"WEB-001","cve_id":"CVE-2021-41773","cvss_score":7.5,"epss_score":0.94,"known_exploited":1,"mitre_technique":"T1190","data_source":"NVD + CISA KEV + EPSS","source_reference":"https://nvd.nist.gov / https://www.cisa.gov/known-exploited-vulnerabilities-catalog / https://www.first.org/epss/"},
        {"asset_id":"WEB-001","cve_id":"CVE-2022-3602","cvss_score":7.5,"epss_score":0.03,"known_exploited":0,"mitre_technique":"T1190","data_source":"NVD + EPSS","source_reference":"https://nvd.nist.gov / https://www.first.org/epss/"},
    ],
    "cti.csv": [
        {"threat_actor":"APT Group","mitre_technique":"T1190","tactic":"Initial Access","target_asset_type":"Web Server","confidence":0.80,"source":"MITRE ATT&CK / CTI feed","first_seen":"2024-01-01","last_seen":"2026-01-01","source_reference":"https://attack.mitre.org/techniques/T1190/"},
        {"threat_actor":"Cybercriminal","mitre_technique":"T1110","tactic":"Credential Access","target_asset_type":"Enterprise App","confidence":0.70,"source":"MITRE ATT&CK / CTI feed","first_seen":"2024-01-01","last_seen":"2026-01-01","source_reference":"https://attack.mitre.org/techniques/T1110/"},
    ],
    "controls.csv": [
        {"asset_id":"WEB-001","control_name":"WAF / virtual patching","control_type":"Preventive","control_coverage":0.70,"control_cost":30000,"mapped_stage":"VII","source_reference":"internal control register"},
        {"asset_id":"DB-001","control_name":"Network segmentation","control_type":"Preventive","control_coverage":0.85,"control_cost":50000,"mapped_stage":"VI/VII","source_reference":"internal control register"},
    ],
    "expert_labels.csv": [
        {"scenario_id":"S001","expert_risk_label":"Critical","expert_risk_score":9.0,"reviewer":"security_expert_1","source_reference":"expert workshop"},
        {"scenario_id":"S002","expert_risk_label":"High","expert_risk_score":7.5,"reviewer":"security_expert_1","source_reference":"expert workshop"},
    ],
    "business_impact.csv": [
        {"asset_id":"WEB-001","business_service":"Customer Portal","revenue_impact_per_hour":15000,"regulatory_impact":0.70,"reputation_impact":0.80,"source_reference":"BIA / risk register"},
        {"asset_id":"DB-001","business_service":"Payment Database","revenue_impact_per_hour":45000,"regulatory_impact":0.95,"reputation_impact":0.95,"source_reference":"BIA / risk register"},
    ],
}


def get_real_data_source_catalog_df():
    return pd.DataFrame(REAL_DATA_SOURCE_CATALOG)


def get_csv_template_df(filename):
    return pd.DataFrame(CSV_TEMPLATE_ROWS.get(filename, []))


def build_template_zip_bytes():
    """Create a ZIP containing all real-data CSV templates and source manifest."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, rows in CSV_TEMPLATE_ROWS.items():
            zf.writestr(fname, pd.DataFrame(rows).to_csv(index=False))
        zf.writestr("official_real_data_sources.json", json.dumps(REAL_DATA_SOURCE_CATALOG, indent=2))
        zf.writestr("README.txt", "Real-data starter pack for PASTA-ML. Use the official source links in official_real_data_sources.json, populate the CSVs, then upload them in the Real Data + CTI tab.\n")
    buf.seek(0)
    return buf.getvalue()


def build_source_manifest_json():
    return json.dumps({
        "purpose": "Official/public real-data source references for PASTA-ML enrichment",
        "usage": "Use these links to obtain real CVE, KEV, EPSS, ATT&CK, STIX/TAXII, SBOM and CVSS data, then upload populated CSV files into the app.",
        "sources": REAL_DATA_SOURCE_CATALOG,
        "templates": list(CSV_TEMPLATE_ROWS.keys()),
    }, indent=2)



def build_builtin_reference_bundle(seed=42):
    """Build a no-upload starter bundle from built-in reference/template rows.

    This is intentionally small and transparent. It lets the app demonstrate the
    real-data workflow immediately without requiring manual CSV upload. Users can
    later replace it with exported CMDB/SBOM/CVE/CTI files.
    """
    raw_assets = get_csv_template_df("assets.csv")
    sbom_df = get_csv_template_df("sbom.csv")
    vulns_df = get_csv_template_df("vulnerabilities.csv")
    cti_df = get_csv_template_df("cti.csv")
    controls_df = get_csv_template_df("controls.csv")
    labels_df = get_csv_template_df("expert_labels.csv")
    business_df = get_csv_template_df("business_impact.csv")
    norm_assets = normalize_uploaded_assets(raw_assets, seed=seed)
    norm_assets, norm_vulns = enrich_assets_with_vulnerabilities(norm_assets, vulns_df, sbom_df)
    return {
        "assets_raw": raw_assets,
        "assets": norm_assets,
        "sbom": sbom_df,
        "vulnerabilities": norm_vulns,
        "cti": cti_df,
        "controls": controls_df,
        "expert_labels": labels_df,
        "business_impact": business_df,
        "bundle_source": "Built-in no-upload reference starter data",
    }


def official_source_markdown_cards():
    """Render official source links as Streamlit-friendly Markdown cards."""
    lines = []
    for item in REAL_DATA_SOURCE_CATALOG:
        lines.append(
            f"**[{item['Source']}]({item['Official URL']})**  \n"
            f"{item['PASTA Stage']} · Suggested file: `{item['Suggested File']}`"
        )
    return "\n\n".join(lines)


def _clean_col(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def read_csv_safely(uploaded_file):
    """Read an uploaded CSV defensively and normalize column names."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [_clean_col(c) for c in df.columns]
        return df
    except Exception as exc:
        st.warning(f"Could not read {getattr(uploaded_file, 'name', 'uploaded file')}: {exc}")
        return pd.DataFrame()


def _coalesce_numeric(df, candidates, default, clip=None):
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").fillna(default)
            if clip is not None:
                s = s.clip(*clip)
            return s
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _coalesce_text(df, candidates, default):
    for c in candidates:
        if c in df.columns:
            return df[c].astype(str).replace({"nan": default, "None": default}).fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def normalize_uploaded_assets(assets_df, seed=42):
    """Convert a real/user asset inventory into the internal asset schema."""
    if assets_df.empty:
        return pd.DataFrame()
    df = assets_df.copy()
    if "asset_id" not in df.columns:
        if "hostname" in df.columns:
            df["asset_id"] = df["hostname"].astype(str)
        elif "name" in df.columns:
            df["asset_id"] = df["name"].astype(str)
        else:
            df["asset_id"] = [f"REAL-{i:04d}" for i in range(len(df))]

    df["asset_type"] = _coalesce_text(df, ["asset_type", "type", "category", "component_type"], "Enterprise App")
    df["criticality"] = _coalesce_numeric(df, ["criticality", "business_criticality", "asset_criticality"], 0.6, (0, 1))
    df["exposure"] = _coalesce_numeric(df, ["exposure", "internet_exposed", "exposure_level"], 0.5, (0, 1))
    # Normalize common Yes/No exposure values if present.
    for c in ["internet_exposed", "public", "external"]:
        if c in df.columns:
            yn = df[c].astype(str).str.lower().map({"yes":1,"true":1,"1":1,"no":0,"false":0,"0":0})
            df["exposure"] = yn.fillna(df["exposure"]).astype(float).clip(0,1)
            break

    df["patch_compliance"] = _coalesce_numeric(df, ["patch_compliance", "patch_status", "patched_ratio"], 0.55, (0, 1))
    df["control_coverage"] = _coalesce_numeric(df, ["control_coverage", "control_effectiveness", "security_control_coverage"], 0.55, (0, 1))
    df["confidentiality_imp"] = _coalesce_numeric(df, ["confidentiality_imp", "confidentiality"], df["criticality"].mean() if len(df) else 0.6, (0, 1))
    df["integrity_imp"] = _coalesce_numeric(df, ["integrity_imp", "integrity"], df["criticality"].mean() if len(df) else 0.6, (0, 1))
    df["availability_imp"] = _coalesce_numeric(df, ["availability_imp", "availability"], df["criticality"].mean() if len(df) else 0.6, (0, 1))
    df["asset_criticality_score"] = ((0.4*df["confidentiality_imp"] + 0.3*df["integrity_imp"] + 0.3*df["availability_imp"]) * df["exposure"]).clip(0,1)
    df["vuln_count"] = _coalesce_numeric(df, ["vuln_count", "vulnerability_count", "cve_count"], 0, (0, 10_000)).astype(int)

    def map_layer(asset_type):
        at = str(asset_type).lower()
        if any(x in at for x in ["database", "db", "scada", "ics", "core"]):
            return "Core"
        if any(x in at for x in ["endpoint", "iot", "user", "camera", "sensor"]):
            return "Access"
        return "Distribution"

    df["layer"] = _coalesce_text(df, ["layer"], "").replace("", np.nan)
    df["layer"] = df["layer"].fillna(df["asset_type"].apply(map_layer))
    df["zone"] = _coalesce_text(df, ["zone", "network_zone"], "").replace("", np.nan)
    df["zone"] = df["zone"].fillna(df["layer"].map(LAYER_ZONES)).fillna("internal")
    df["layer_ord"] = df["layer"].map(LAYER_ORDINAL).fillna(1).astype(int)

    keep = ["asset_id", "asset_type", "criticality", "exposure", "patch_compliance", "control_coverage",
            "confidentiality_imp", "integrity_imp", "availability_imp", "asset_criticality_score",
            "vuln_count", "layer", "zone", "layer_ord"]
    return df[keep].drop_duplicates("asset_id").reset_index(drop=True)


def enrich_assets_with_vulnerabilities(asset_df, vulns_df, sbom_df=None):
    """Aggregate CVE/SBOM data into asset-level fields while preserving detail tables."""
    if asset_df.empty:
        return asset_df, pd.DataFrame()
    assets = asset_df.copy()
    vulns = vulns_df.copy() if vulns_df is not None else pd.DataFrame()
    sbom = sbom_df.copy() if sbom_df is not None else pd.DataFrame()

    if vulns.empty and not sbom.empty and "cve_id" in sbom.columns:
        vulns = sbom.copy()

    if not vulns.empty:
        if "asset_id" not in vulns.columns and not sbom.empty and "component_name" in vulns.columns and "component_name" in sbom.columns and "asset_id" in sbom.columns:
            vulns = vulns.merge(sbom[["component_name", "asset_id"]].drop_duplicates(), on="component_name", how="left")
        if "asset_id" in vulns.columns:
            vulns["cvss_score"] = _coalesce_numeric(vulns, ["cvss_score", "cvss", "base_score"], 5.0, (0,10))
            vulns["epss_score"] = _coalesce_numeric(vulns, ["epss_score", "epss", "epss_probability"], 0.2, (0,1))
            vulns["known_exploited"] = _coalesce_numeric(vulns, ["known_exploited", "kev", "cisa_kev"], 0, (0,1)).astype(int)
            agg = vulns.groupby("asset_id").agg(
                vuln_count_real=("cvss_score", "size"),
                cvss_weighted_avg_real=("cvss_score", "mean"),
                max_cvss_real=("cvss_score", "max"),
                epss_max_real=("epss_score", "max"),
                known_exploited_count=("known_exploited", "sum"),
            ).reset_index()
            assets = assets.merge(agg, on="asset_id", how="left")
            for c in ["vuln_count_real", "known_exploited_count"]:
                assets[c] = assets[c].fillna(0).astype(int)
            for c in ["cvss_weighted_avg_real", "max_cvss_real", "epss_max_real"]:
                assets[c] = assets[c].fillna(0.0)
            # Prefer real CVE count when available, otherwise keep original count.
            assets["vuln_count"] = np.where(assets["vuln_count_real"] > 0, assets["vuln_count_real"], assets["vuln_count"])
    return assets, vulns


def build_real_environment_from_uploads(assets_df, actors_df=None, seed=42):
    """Build an environment object from uploaded assets, including topology and centrality."""
    if assets_df.empty:
        return None
    asset_df = assets_df.copy()
    ids_by_layer = {layer: asset_df.loc[asset_df["layer"] == layer, "asset_id"].astype(str).tolist()
                    for layer in ["Core", "Distribution", "Access"]}
    topology_json = build_layered_topology(json.dumps(ids_by_layer), int(seed))
    cent_df = compute_centrality_features(topology_json)
    asset_df = asset_df.drop(columns=[c for c in CENTRALITY_FEATS if c in asset_df.columns], errors="ignore")
    asset_df = asset_df.merge(cent_df[["asset_id"] + CENTRALITY_FEATS], on="asset_id", how="left")
    for f in CENTRALITY_FEATS:
        asset_df[f] = asset_df[f].fillna(0.0)
    if actors_df is None or actors_df.empty:
        actors = []
        for ta_type, cfg in THREAT_ACTORS.items():
            actors.append({"actor_type": ta_type, "capability": np.mean(cfg["capability"])/10, "persistence": np.mean(cfg["persistence"])/10, "motivation": cfg["motivation"], "n_techniques": 5})
        actor_df = pd.DataFrame(actors)
    else:
        actor_df = actors_df.copy()
    return {"assets": asset_df, "actors": actor_df, "seed": seed, "n_assets": len(asset_df), "n_actors": len(actor_df), "topology_json": topology_json, "mode": "real-data"}


def compute_probabilistic_attack_paths(asset_df, topology_json, top_k=10):
    """Rank probable/high-impact attack paths using edge/node probabilities."""
    if asset_df is None or asset_df.empty or not topology_json:
        return pd.DataFrame()
    G = nx.node_link_graph(json.loads(topology_json))
    lookup = asset_df.set_index("asset_id").to_dict("index")
    entries = asset_df.nlargest(max(3, len(asset_df)//10), "exposure")["asset_id"].astype(str).tolist()
    targets = asset_df.nlargest(max(3, len(asset_df)//10), "asset_criticality_score")["asset_id"].astype(str).tolist()
    rows = []
    for src in entries:
        for tgt in targets:
            if src == tgt or src not in G or tgt not in G:
                continue
            try:
                paths = list(nx.shortest_simple_paths(G, src, tgt))[:3]
            except Exception:
                continue
            for path in paths:
                probs = []
                impact = 0.0
                control_fail = []
                for node in path:
                    a = lookup.get(node, {})
                    p = (0.30 * float(a.get("exposure", 0.5)) +
                         0.25 * (1 - float(a.get("patch_compliance", 0.5))) +
                         0.25 * min(float(a.get("vuln_count", 0))/30.0, 1.0) +
                         0.20 * float(a.get("betweenness_centrality", 0.0)))
                    probs.append(np.clip(p, 0.01, 0.95))
                    control_fail.append(1 - float(a.get("control_coverage", 0.5)))
                    impact = max(impact, float(a.get("asset_criticality_score", 0.5)))
                compromise_prob = float(np.prod(probs) * np.mean(control_fail))
                business_risk = float(compromise_prob * impact * 10)
                rows.append({"source": src, "target": tgt, "path": " → ".join(map(str, path)),
                             "path_length": len(path), "compromise_probability": round(compromise_prob, 4),
                             "business_impact": round(impact, 3), "path_risk_score": round(business_risk, 3)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["path_risk_score", "compromise_probability"], ascending=False).head(top_k).reset_index(drop=True)


def compute_fair_results(asset_df, features_df=None, asset_value_default=100000.0, control_cost_default=15000.0):
    """FAIR-style lightweight financial risk quantification."""
    if asset_df is None or asset_df.empty:
        return pd.DataFrame()
    df = asset_df.copy()
    df["asset_value"] = _coalesce_numeric(df, ["asset_value", "business_value", "replacement_value"], asset_value_default, (0, 1e12))
    df["loss_event_frequency"] = (0.2 + 2.0*df["exposure"].astype(float) + 1.5*(1-df["patch_compliance"].astype(float))).clip(0.05, 12)
    df["vulnerability_probability"] = (0.2 + 0.5*(1-df["control_coverage"].astype(float)) + 0.3*np.minimum(df["vuln_count"].astype(float)/30.0, 1)).clip(0.01, 0.95)
    df["primary_loss"] = df["asset_value"] * (0.10 + 0.60*df["asset_criticality_score"].astype(float))
    df["secondary_loss"] = df["primary_loss"] * (0.10 + 0.40*df["exposure"].astype(float))
    df["annualized_loss_expectancy"] = df["loss_event_frequency"] * df["vulnerability_probability"] * (df["primary_loss"] + df["secondary_loss"])
    df["control_cost"] = control_cost_default
    df["expected_risk_reduction"] = (0.25 + 0.45*(1-df["control_coverage"].astype(float))).clip(0.05, 0.75)
    df["residual_financial_exposure"] = df["annualized_loss_expectancy"] * (1-df["expected_risk_reduction"])
    df["control_roi"] = (df["annualized_loss_expectancy"] - df["residual_financial_exposure"] - df["control_cost"]) / df["control_cost"].replace(0, np.nan)
    cols = ["asset_id", "asset_type", "asset_value", "loss_event_frequency", "vulnerability_probability", "annualized_loss_expectancy", "expected_risk_reduction", "residual_financial_exposure", "control_cost", "control_roi"]
    return df[cols].sort_values("annualized_loss_expectancy", ascending=False).reset_index(drop=True)


def assess_mm_pasta(process_formalization, tooling_integration, automation_depth, scalability_outcome, model_freshness, risk_ticket_conversion, coverage):
    vals = [process_formalization, tooling_integration, automation_depth, scalability_outcome, model_freshness, risk_ticket_conversion, coverage]
    score = float(np.mean(vals))
    if score < 25: level, name = 1, "Ad-hoc workshops"
    elif score < 45: level, name = 2, "Template-based periodic PASTA"
    elif score < 65: level, name = 3, "Release-gate integrated"
    elif score < 85: level, name = 4, "Metric-driven continuous PASTA"
    else: level, name = 5, "AI-assisted continuous PASTA"
    recs = []
    if tooling_integration < 60: recs.append("Integrate SBOM, cloud inventory, vulnerability scanner, CTI and ticketing sources.")
    if automation_depth < 60: recs.append("Automate Stage II/V ingestion and Stage IV ATT&CK mapping first; Stage VI next.")
    if model_freshness < 60: recs.append("Add drift detection and trigger reassessment on architecture/CVE changes.")
    if risk_ticket_conversion < 60: recs.append("Export high-risk findings as Jira/GitHub/ServiceNow-ready backlog items.")
    if coverage < 60: recs.append("Prioritize crown-jewel and internet-facing systems, then expand portfolio coverage.")
    return {"score": round(score,1), "level": level, "level_name": name, "recommendations": recs}


def compute_freshness_and_drift(current_assets, previous_assets=None, last_update_date=None, current_vulns=None, previous_vulns=None):
    today = datetime.now().date()
    days_stale = 0
    if last_update_date:
        try:
            days_stale = max(0, (today - pd.to_datetime(last_update_date).date()).days)
        except Exception:
            days_stale = 0
    cur_ids = set(current_assets["asset_id"].astype(str)) if current_assets is not None and not current_assets.empty and "asset_id" in current_assets else set()
    prev_ids = set(previous_assets["asset_id"].astype(str)) if previous_assets is not None and not previous_assets.empty and "asset_id" in previous_assets else set()
    new_assets = len(cur_ids - prev_ids) if prev_ids else 0
    removed_assets = len(prev_ids - cur_ids) if prev_ids else 0
    asset_delta_pct = safe_div(new_assets + removed_assets, max(len(cur_ids), 1), 0) * 100
    cur_cves = set(current_vulns["cve_id"].astype(str)) if current_vulns is not None and not current_vulns.empty and "cve_id" in current_vulns else set()
    prev_cves = set(previous_vulns["cve_id"].astype(str)) if previous_vulns is not None and not previous_vulns.empty and "cve_id" in previous_vulns else set()
    new_cves = len(cur_cves - prev_cves) if prev_cves else 0
    freshness_score = max(0, 100 - min(days_stale*2, 50) - min(asset_delta_pct, 30) - min(new_cves*2, 20))
    return {"days_since_last_update": int(days_stale), "new_assets": int(new_assets), "removed_assets": int(removed_assets), "asset_delta_pct": round(asset_delta_pct,2), "new_cves": int(new_cves), "model_freshness_score": round(float(freshness_score),1), "reassessment_recommended": bool(freshness_score < 70 or new_cves > 0 or asset_delta_pct > 10)}


def create_ticket_backlog(mitigation_df=None, fair_df=None, features_df=None):
    """Create Jira/GitHub/ServiceNow-ready remediation backlog."""
    rows = []
    if mitigation_df is not None and not mitigation_df.empty:
        for i, r in mitigation_df.head(100).iterrows():
            risk = float(r.get("risk_score", r.get("residual_risk_score", 5)))
            priority = "P1" if risk >= 8 else "P2" if risk >= 6 else "P3" if risk >= 4 else "P4"
            sla = "7 days" if priority == "P1" else "14 days" if priority == "P2" else "30 days" if priority == "P3" else "90 days"
            rows.append({"ticket_id": f"PASTA-{i+1:04d}", "asset_id": r.get("asset_id", "N/A"), "finding": r.get("rationale", "High PASTA risk finding"), "recommended_action": r.get("recommended_action", r.get("mitigation_action", "Review and remediate risk")), "pasta_stage": r.get("pasta_stage", "VII"), "risk_score": round(risk,2), "priority": priority, "sla": sla, "owner": r.get("owner", "Security/Platform Team"), "residual_risk": r.get("residual_risk_score", "TBD")})
    elif fair_df is not None and not fair_df.empty:
        for i, r in fair_df.head(100).iterrows():
            ale = float(r.get("annualized_loss_expectancy", 0))
            priority = "P1" if ale >= 250000 else "P2" if ale >= 100000 else "P3" if ale >= 25000 else "P4"
            rows.append({"ticket_id": f"PASTA-{i+1:04d}", "asset_id": r.get("asset_id", "N/A"), "finding": "High expected financial exposure", "recommended_action": "Apply prioritized control package and validate residual exposure", "pasta_stage": "VII", "risk_score": round(min(10, ale/100000),2), "priority": priority, "sla": "14 days" if priority in ["P1","P2"] else "30 days", "owner": "Security/GRC Team", "residual_risk": round(float(r.get("residual_financial_exposure", 0)),2)})
    return pd.DataFrame(rows)


def build_pif_export(session_state, experiment_config=None, max_records=500):
    """Build PASTA Interchange Format (PIF) JSON across all seven stages."""
    def df_records(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.head(max_records).replace({np.nan: None}).to_dict("records")
        return []
    env = session_state.get("env")
    bundle = session_state.get("real_data_bundle") or {}
    pif = {
        "pif_version": PIF_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "metadata": {"framework": "PASTA-ML", "mode": env.get("mode", "simulation") if isinstance(env, dict) else "unknown", "experiment_config": experiment_config or {}},
        "stage_1_business_objectives": {"business_impact": df_records(bundle.get("business_impact", pd.DataFrame())), "maturity": session_state.get("maturity_results")},
        "stage_2_assets_scope": {"assets": df_records(env["assets"] if isinstance(env, dict) and "assets" in env else pd.DataFrame()), "sbom": df_records(bundle.get("sbom", pd.DataFrame()))},
        "stage_3_architecture_decomposition": {"topology_node_link": json.loads(env.get("topology_json", "{}")) if isinstance(env, dict) and env.get("topology_json") else {}, "freshness": session_state.get("freshness_results")},
        "stage_4_threat_analysis": {"threat_actors": df_records(env["actors"] if isinstance(env, dict) and "actors" in env else pd.DataFrame()), "cti": df_records(bundle.get("cti", pd.DataFrame()))},
        "stage_5_vulnerability_analysis": {"vulnerabilities": df_records(bundle.get("vulnerabilities", pd.DataFrame()))},
        "stage_6_attack_simulation": {"scenarios": df_records(session_state.get("scenarios", pd.DataFrame())), "monte_carlo_stats": session_state.get("mc_stats"), "probabilistic_paths": df_records(session_state.get("prob_paths", pd.DataFrame()))},
        "stage_7_risk_impact": {"features": df_records(session_state.get("features", pd.DataFrame())), "fair": df_records(session_state.get("fair_results", pd.DataFrame())), "tickets": df_records(session_state.get("ticket_backlog", pd.DataFrame())), "review_log": df_records(session_state.get("review_log", pd.DataFrame()))},
    }
    return pif


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
    st.markdown("### 🚨 Step 5b — Alerting (Monte Carlo)")
    st.caption("Stochastic attacker simulation + binary classifier (attack vs normal).")
    mc_n_sims     = st.slider("Monte-Carlo simulations", 10, 500, 100, 10)
    mc_steps      = st.slider("Max attack steps per sim", 4, 40, 12)
    mc_epsilon    = st.slider("ε (exploration prob.)", 0.0, 0.6, 0.25, 0.05,
                              help="0 = pure greedy attacker, higher = more stochastic exploration")
    mc_norm_alert = st.slider("Normal-traffic false-alarm rate", 0.0, 0.20, 0.05, 0.01)

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
 tab_step4, tab_step5, tab_step5b, tab_step6,
 tab_realdata, tab_ops, tab_export) = st.tabs([
    "🏠 Overview",
    "📐 Step 1 · Framework",
    "🏗️ Step 2 · Environment",
    "🎲 Step 3 · Scenarios",
    "🔧 Step 4 · Features",
    "🤖 Step 5 · ML Models",
    "🚨 Step 5b · Alerting",
    "⚡ Step 6 · Scalability",
    "🧩 Real Data + CTI",
    "🏛️ Ops + Governance",
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

        # ── NEW: Layered Network Topology (BA + WS + Tree) ──────────────────
        st.markdown("#### 🧱 Layered Enterprise Topology (Core / Distribution / Access)")
        st.markdown(
            "<div class='callout-info'>The enterprise graph is composed of three "
            "layers built from different random-graph models that match each layer's "
            "empirical character: <b>Core</b> = Barabási–Albert (scale-free hubs), "
            "<b>Distribution</b> = Watts–Strogatz (small-world), <b>Access</b> = "
            "random tree. Inter-layer wiring runs Access → Distribution → Core "
            "(attack-flow direction).</div>",
            unsafe_allow_html=True)

        G_topo = nx.node_link_graph(json.loads(env["topology_json"]))
        # Layout: spring on the undirected view for visual clarity
        pos = nx.spring_layout(G_topo.to_undirected(), seed=42, k=0.9, iterations=80)

        edge_x, edge_y = [], []
        for u, v in G_topo.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                line=dict(color="#cccccc", width=0.7),
                                hoverinfo="none", showlegend=False)

        node_traces = [edge_trace]
        for layer_name, color in LAYER_COLORS.items():
            xs, ys, labels = [], [], []
            for n, d in G_topo.nodes(data=True):
                if d.get("layer") == layer_name:
                    xs.append(pos[n][0]); ys.append(pos[n][1]); labels.append(n)
            if xs:
                node_traces.append(go.Scatter(
                    x=xs, y=ys, mode="markers", name=f"{layer_name} ({len(xs)})",
                    marker=dict(size=11, color=color,
                                line=dict(color="white", width=1.2)),
                    text=labels,
                    hovertemplate="<b>%{text}</b><br>Layer: " + layer_name +
                                  "<extra></extra>",
                ))

        fig_topo = go.Figure(node_traces)
        fig_topo.update_layout(
            title="Layered Network Topology — node size & colour by layer",
            showlegend=True, height=520,
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            plot_bgcolor="#0f1923", paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig_topo, use_container_width=True)

        # Centrality distributions per layer
        st.markdown("##### 📊 Graph-Structural (Centrality) Features by Layer")
        st.caption("These features complement the asset-intrinsic features and "
                   "are fed into the Step 5b alerting classifier.")
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_dc = px.box(asset_df, x="layer", y="degree_centrality",
                            color="layer", color_discrete_map=LAYER_COLORS,
                            category_orders={"layer": ["Access", "Distribution", "Core"]},
                            title="Degree Centrality by Layer")
            fig_dc.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                  showlegend=False)
            st.plotly_chart(fig_dc, use_container_width=True)
        with gc2:
            fig_bc = px.box(asset_df, x="layer", y="betweenness_centrality",
                            color="layer", color_discrete_map=LAYER_COLORS,
                            category_orders={"layer": ["Access", "Distribution", "Core"]},
                            title="Betweenness Centrality by Layer")
            fig_bc.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                  showlegend=False)
            st.plotly_chart(fig_bc, use_container_width=True)

        gc3, gc4 = st.columns(2)
        with gc3:
            fig_ec = px.box(asset_df, x="layer", y="eigenvector_centrality",
                            color="layer", color_discrete_map=LAYER_COLORS,
                            category_orders={"layer": ["Access", "Distribution", "Core"]},
                            title="Eigenvector Centrality by Layer")
            fig_ec.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                  showlegend=False)
            st.plotly_chart(fig_ec, use_container_width=True)
        with gc4:
            fig_cl = px.box(asset_df, x="layer", y="clustering_coefficient",
                            color="layer", color_discrete_map=LAYER_COLORS,
                            category_orders={"layer": ["Access", "Distribution", "Core"]},
                            title="Clustering Coefficient by Layer")
            fig_cl.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                  showlegend=False)
            st.plotly_chart(fig_cl, use_container_width=True)

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
                    env["topology_json"],
                    n_scenarios, tuple(selected_vecs), rng_seed, max_path_len)
                st.session_state["scenarios"] = sc_df
                st.session_state["attack_graph"] = g_data
                st.session_state["topology"] = env["topology_json"]

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

            # Risk score design and mitigation outputs
            st.markdown("#### 📐 Risk Target Design — Baseline vs Hybrid Target")
            st.markdown("""
<div class='formula-box'>
Baseline_Risk = transparent weighted PASTA formula<br>
Outcome_Risk = EPSS/CVSS + reachability + exploit maturity + control bypass factors<br>
Risk_Score = 0.55·Baseline_Risk + 0.35·Outcome_Risk + 0.10·Impact + calibrated heterogeneity<br>
Residual_Risk = Risk_Score × (1 − recommended_control_reduction)
</div>
""", unsafe_allow_html=True)

            st.markdown("#### 🛡️ Stage-7 Mitigation Recommendations")
            st.caption("Each scenario now includes action, mapped PASTA stage, rationale, reduction factor, and residual risk.")
            mit_cols = ["risk_score", "residual_risk_score", "mitigation_actions", "mitigation_stages", "mitigation_rationale"]
            st.dataframe(feat_df[mit_cols].sort_values("risk_score", ascending=False).head(20),
                         use_container_width=True, hide_index=True)

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
        "R², MAE, RMSE, MAPE, k-fold cross-validation, grouped holdout validation, "
        "uncertainty diagnostics, ablation analysis, SHAP explainability, and "
        "permutation-based feature importance.</div>", unsafe_allow_html=True)

    feat_ready = st.session_state["features"] is not None
    if not feat_ready:
        st.warning("⚠️ Please complete **Step 4 — Feature Engineering** first.")
    else:
        if st.button("▶ Train & Evaluate ML Models", type="primary", key="run_ml"):
            feat_df = st.session_state["features"]
            # Pass the full feature frame so grouped validation and formula baseline can use
            # asset_type / attack_vector / baseline_risk_score traceability columns.
            feat_json = feat_df.to_json(orient="records")
            with st.spinner("Training baselines + ML models… computing SHAP and grouped validation…"):
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
                    "Asset-Type Holdout R²": res.get("group_asset_r2", np.nan),
                    "Attack-Vector Holdout R²": res.get("group_vector_r2", np.nan),
                    "MITRE-Technique Holdout R²": res.get("group_mitre_r2", np.nan),
                    "Target/Baseline Corr": res.get("target_baseline_corr", np.nan),
                    "Uncertainty P90 Width": res.get("uncertainty_p90_width", np.nan),
                    "Kind": res.get("model_kind", "ml"),
                    "Train Time (s)": res["train_time_s"],
                    "Infer Time (ms)": res["infer_ms"],
                    "Train N": res["n_train"],
                    "Test N": res["n_test"],
                })
            mdf = pd.DataFrame(metric_rows)
            st.dataframe(mdf.set_index("Model"), use_container_width=True)

            # Dataset diagnostics and feature-family ablation. This is useful for
            # thesis defence because it shows how much the target still depends on
            # the transparent PASTA baseline versus outcome-inspired signals.
            first_res = next(iter(ml_res.values()))
            diag_cols = st.columns(2)
            diag_cols[0].metric("Target ↔ PASTA Baseline Corr.", f"{first_res.get('target_baseline_corr', np.nan):.3f}")
            diag_cols[1].metric("Target ↔ Outcome Risk Corr.", f"{first_res.get('target_outcome_corr', np.nan):.3f}")
            if first_res.get("ablation_rows"):
                st.markdown("#### 🧪 Feature-Family Ablation")
                st.dataframe(pd.DataFrame(first_res["ablation_rows"]).set_index("Feature Group"), use_container_width=True)

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
# TAB: STEP 5b — Monte-Carlo Alerting Classifier  (NEW)
# ═══════════════════════════════════════════════════════════════════════════
with tab_step5b:
    st.subheader("🚨 Step 5b: Monte-Carlo Alerting Classifier")
    st.markdown(
        "<div class='callout-info'>Runs a stochastic <b>ε-greedy attacker</b> "
        "across the layered enterprise topology K times, producing a labelled "
        "event-level dataset (attack vs normal). A binary classifier is then "
        "trained on graph-structural + asset-intrinsic features to predict "
        "<b>alert/no-alert</b> — the operationally relevant task in security "
        "monitoring. Unlike Step 5 (regression), the label here is grounded in "
        "<b>actual simulated compromise</b>, not the formula-derived risk score, "
        "so the two heads are genuinely complementary.</div>",
        unsafe_allow_html=True)

    env_ready = st.session_state["env"] is not None
    if not env_ready:
        st.warning("⚠️ Please run **Step 2 — Environment Simulation** first.")
    else:
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("▶ Run Monte-Carlo Attack Simulation", type="primary",
                         key="run_mc"):
                env = st.session_state["env"]
                with st.spinner(f"Running {mc_n_sims} ε-greedy attack simulations…"):
                    ev_df, paths, stats = monte_carlo_attack_simulation(
                        env["assets"].to_json(orient="records"),
                        env["topology_json"],
                        int(mc_n_sims), int(mc_steps),
                        float(mc_epsilon), int(rng_seed),
                        float(mc_norm_alert))
                    st.session_state["mc_events"] = ev_df
                    st.session_state["mc_paths"]  = paths
                    st.session_state["mc_stats"]  = stats
        with bc2:
            mc_ready = st.session_state["mc_events"] is not None
            train_btn_disabled = not mc_ready
            if st.button("▶ Train Alerting Classifier", type="primary",
                         key="run_clf", disabled=train_btn_disabled):
                with st.spinner("Training Random Forest + Gradient Boosting "
                                "classifiers + 5-fold CV…"):
                    clf_res = train_alert_classifier(
                        st.session_state["mc_events"].to_json(orient="records"),
                        test_size, cv_folds)
                    st.session_state["clf_results"] = clf_res

        if st.session_state["mc_events"] is None:
            st.info("👆 Step 1 of 2: Click **Run Monte-Carlo Attack Simulation**.")
        else:
            ev_df = st.session_state["mc_events"]
            stats = st.session_state["mc_stats"]
            paths = st.session_state["mc_paths"]

            # KPIs from Monte-Carlo simulation
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Simulations",         stats["n_simulations"])
            m2.metric("Mean Path Length",    f"{stats['mean_path_length']:.2f}")
            m3.metric("Std Path Length",     f"{stats['std_path_length']:.2f}")
            m4.metric("Unique Compromised",  stats["unique_assets_compromised"])
            m5.metric("Core Breach Rate",    f"{stats['core_compromise_rate']*100:.1f}%")

            # Path-length distribution + per-layer compromise rate
            pc1, pc2 = st.columns(2)
            with pc1:
                pl = [len(p) for p in paths]
                fig_pl = px.histogram(x=pl, nbins=max(5, min(30, len(set(pl)))),
                                       color_discrete_sequence=["#8e44ad"],
                                       title="Attack-Path Length Distribution "
                                             f"(N={len(paths)} sims)")
                fig_pl.add_vline(x=stats["mean_path_length"], line_dash="dash",
                                 annotation_text=f"mean={stats['mean_path_length']:.2f}",
                                 annotation_position="top right")
                fig_pl.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0),
                                      xaxis_title="Path length (hops)",
                                      yaxis_title="Frequency")
                st.plotly_chart(fig_pl, use_container_width=True)
            with pc2:
                # Compromise rate per layer
                compromised_set = set().union(*[set(p) for p in paths]) if paths else set()
                env = st.session_state["env"]
                layer_counts = (
                    env["assets"][["asset_id", "layer"]]
                    .assign(compromised=lambda d: d["asset_id"].isin(compromised_set))
                    .groupby("layer")["compromised"].agg(["sum", "count"])
                    .reset_index()
                )
                layer_counts["rate_%"] = (layer_counts["sum"] / layer_counts["count"]) * 100
                layer_counts = layer_counts.sort_values(
                    "layer", key=lambda x: x.map({"Access":0,"Distribution":1,"Core":2}))
                fig_lc = px.bar(layer_counts, x="layer", y="rate_%",
                                color="layer", color_discrete_map=LAYER_COLORS,
                                title="Asset Compromise Rate by Layer")
                fig_lc.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0),
                                      showlegend=False,
                                      yaxis_title="% of layer compromised",
                                      xaxis_title="Layer")
                st.plotly_chart(fig_lc, use_container_width=True)

            # Class balance + event preview
            cb1, cb2 = st.columns([1, 2])
            with cb1:
                lbl_counts = ev_df["label"].value_counts().reset_index()
                lbl_counts.columns = ["Label", "Count"]
                fig_lb = px.pie(lbl_counts, names="Label", values="Count",
                                hole=0.45,
                                color="Label",
                                color_discrete_map={"attack":"#c0392b",
                                                     "normal":"#27ae60"},
                                title="Event Class Balance")
                fig_lb.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_lb, use_container_width=True)
            with cb2:
                st.markdown("##### 📋 Event-Level Dataset (first 20 rows)")
                st.dataframe(ev_df[["simulation","step","asset_id","layer",
                                     "asset_type","label","alert",
                                     "criticality","exposure",
                                     "betweenness_centrality"]].head(20),
                             use_container_width=True)

            st.download_button("📥 Download Event Dataset (CSV)",
                               ev_df.to_csv(index=False).encode(),
                               "mc_event_dataset.csv", "text/csv")

            # ── Classifier results ─────────────────────────────────────────
            if st.session_state["clf_results"] is None:
                st.info("👆 Step 2 of 2: Click **Train Alerting Classifier**.")
            else:
                clf_res = st.session_state["clf_results"]
                if "error" in clf_res:
                    st.error(clf_res["error"])
                else:
                    st.markdown("#### 📊 Classifier Comparison — Operational Metrics")
                    rows = []
                    for name, r in clf_res.items():
                        rows.append({
                            "Model":      name,
                            "Accuracy":   r["accuracy"],
                            "Precision":  r["precision"],
                            "Recall":     r["recall"],
                            "F1":         r["f1"],
                            "ROC-AUC":    r["roc_auc"],
                            "PR-AUC":     r["pr_auc"],
                            f"CV F1 ({cv_folds}-fold)":
                                f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}",
                            "Train (s)":  r["train_time_s"],
                            "Infer (ms)": r["infer_ms"],
                            "n_train":    r["n_train"],
                            "n_test":     r["n_test"],
                        })
                    st.dataframe(pd.DataFrame(rows).set_index("Model"),
                                 use_container_width=True)

                    # Best-model badge
                    best_f1 = max(r["f1"] for r in clf_res.values())
                    if   best_f1 >= 0.90: badge = "🟢 Excellent (F1 ≥ 0.90)"
                    elif best_f1 >= 0.80: badge = "🟡 Good (F1 ≥ 0.80)"
                    else:                  badge = "🔴 Needs tuning (F1 < 0.80)"
                    st.markdown(
                        f"<div class='callout-good'>Best F1: <b>{best_f1:.4f}</b> — "
                        f"{badge}</div>", unsafe_allow_html=True)

                    # Per-model diagnostics
                    sub_tabs = st.tabs(list(clf_res.keys()))
                    for tab_c, (name, r) in zip(sub_tabs, clf_res.items()):
                        with tab_c:
                            d1, d2 = st.columns(2)
                            with d1:
                                # Confusion matrix
                                cm = np.array(r["confusion"])
                                fig_cm = px.imshow(
                                    cm, text_auto=True, aspect="equal",
                                    x=["Pred Normal","Pred Attack"],
                                    y=["True Normal","True Attack"],
                                    color_continuous_scale="Blues",
                                    title=f"{name}: Confusion Matrix")
                                fig_cm.update_layout(height=320,
                                    margin=dict(l=0,r=0,t=40,b=0))
                                st.plotly_chart(fig_cm, use_container_width=True)
                            with d2:
                                # ROC curve
                                y_t  = np.array(r["y_test"])
                                y_pp = np.array(r["y_proba"])
                                try:
                                    fpr, tpr, _ = roc_curve(y_t, y_pp)
                                except Exception:
                                    fpr, tpr = np.array([0,1]), np.array([0,1])
                                fig_roc = go.Figure()
                                fig_roc.add_trace(go.Scatter(
                                    x=fpr, y=tpr, mode="lines",
                                    name=f"ROC (AUC={r['roc_auc']:.3f})",
                                    line=dict(color="#2980b9", width=2)))
                                fig_roc.add_trace(go.Scatter(
                                    x=[0,1], y=[0,1], mode="lines",
                                    name="Random",
                                    line=dict(color="grey", dash="dash")))
                                fig_roc.update_layout(
                                    title=f"{name}: ROC Curve",
                                    xaxis_title="False Positive Rate",
                                    yaxis_title="True Positive Rate",
                                    height=320, margin=dict(l=0,r=0,t=40,b=0),
                                    legend=dict(y=0.05, x=0.55))
                                st.plotly_chart(fig_roc, use_container_width=True)

                            # PR curve + Feature importance
                            d3, d4 = st.columns(2)
                            with d3:
                                try:
                                    prec, rec, _ = precision_recall_curve(y_t, y_pp)
                                except Exception:
                                    prec, rec = np.array([0,1]), np.array([1,0])
                                fig_pr = go.Figure()
                                fig_pr.add_trace(go.Scatter(
                                    x=rec, y=prec, mode="lines",
                                    name=f"PR (AP={r['pr_auc']:.3f})",
                                    line=dict(color="#c0392b", width=2)))
                                fig_pr.update_layout(
                                    title=f"{name}: Precision–Recall",
                                    xaxis_title="Recall",
                                    yaxis_title="Precision",
                                    height=320, margin=dict(l=0,r=0,t=40,b=0),
                                    legend=dict(y=0.05, x=0.55))
                                st.plotly_chart(fig_pr, use_container_width=True)
                            with d4:
                                imp = np.array(r["feat_imp"])
                                fn  = r["feat_names"]
                                order = np.argsort(imp)
                                fig_fi = go.Figure(go.Bar(
                                    y=[fn[i] for i in order],
                                    x=imp[order], orientation="h",
                                    marker_color="#8e44ad"))
                                fig_fi.update_layout(
                                    title=f"{name}: Feature Importance",
                                    xaxis_title="Importance",
                                    height=320, margin=dict(l=0,r=0,t=40,b=0))
                                st.plotly_chart(fig_fi, use_container_width=True)

                    st.markdown(
                        "<div class='callout-warn'>"
                        "<b>Methodological note for thesis defence:</b> "
                        "The alerting label is derived from <i>actual simulated "
                        "compromise</i> by the ε-greedy attacker, not from the "
                        "regression target (risk_score). This means Steps 5 and "
                        "5b are genuinely complementary tasks rather than two "
                        "views of the same formula — a common subtle leakage "
                        "issue in synthetic security datasets.</div>",
                        unsafe_allow_html=True)

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

# ═══════════════════════════════════════════════════════════════════════════
# TAB: REAL DATA + CTI — Uploads, SBOM/CVE correlation, probabilistic paths
# ═══════════════════════════════════════════════════════════════════════════
with tab_realdata:
    st.subheader("🧩 Real Data + CTI Enrichment")
    st.caption("Use built-in hyperlinks/reference data, or upload real/anonymized asset, SBOM, CVE, CTI and label files. If nothing is uploaded, the simulation pipeline remains unchanged.")

    st.markdown("#### 🔗 One-click real-data source references")
    st.markdown("The official data-source hyperlinks are embedded directly in the app. You can open them anytime without searching or preparing a separate document.")

    with st.container(border=True):
        st.markdown("##### Official links")
        st.markdown(official_source_markdown_cards())

    src_df = get_real_data_source_catalog_df()
    st.dataframe(
        src_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Official URL": st.column_config.LinkColumn("Official URL"),
            "Use in App": st.column_config.TextColumn("Use in App", width="large"),
        },
    )

    st.markdown("##### No-upload quick start")
    st.caption("Use this when you want the app to immediately populate the real-data workflow from the built-in reference starter dataset. Replace it later with exported real CMDB/SBOM/CVE/CTI files when available.")
    quick_cols = st.columns(3)
    with quick_cols[0]:
        if st.button("⚡ Load built-in reference data", key="load_builtin_reference_bundle_v5"):
            bundle_ref = build_builtin_reference_bundle(seed=rng_seed)
            st.session_state["real_data_bundle"] = bundle_ref
            env_ref = build_real_environment_from_uploads(bundle_ref["assets"], seed=rng_seed)
            if env_ref:
                st.session_state["env"] = env_ref
                st.session_state["topology"] = env_ref["topology_json"]
            st.success("Loaded built-in reference data. The app is now ready without CSV upload.")
    with quick_cols[1]:
        st.download_button(
            "📦 Download templates",
            build_template_zip_bytes(),
            "pasta_real_data_starter_pack.zip",
            "application/zip",
            key="download_real_data_pack_v5",
        )
    with quick_cols[2]:
        st.download_button(
            "🧾 Download source manifest",
            build_source_manifest_json().encode(),
            "official_real_data_sources.json",
            "application/json",
            key="download_source_manifest_v5",
        )

    with st.expander("🌐 Direct source URLs for API/browser use", expanded=False):
        st.code("""NVD CVE API: https://nvd.nist.gov/developers/vulnerabilities
CISA KEV CSV: https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv
FIRST EPSS API: https://api.first.org/data/v1/epss
MITRE ATT&CK STIX: https://github.com/mitre-attack/attack-stix-data
MITRE ATT&CK Browser: https://attack.mitre.org/
CycloneDX SBOM: https://cyclonedx.org/specification/overview/
SPDX SBOM: https://spdx.dev/
CVSS: https://www.first.org/cvss/""", language="text")

    with st.expander("📄 Preview/download individual CSV templates", expanded=False):
        template_cols = st.columns(2)
        for idx, fname in enumerate(CSV_TEMPLATE_ROWS.keys()):
            with template_cols[idx % 2]:
                tdf = get_csv_template_df(fname)
                st.markdown(f"**{fname}**")
                st.dataframe(tdf, use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {fname}",
                    tdf.to_csv(index=False).encode(),
                    fname,
                    "text/csv",
                    key=f"download_template_{fname}_v4",
                )

    st.divider()

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        assets_file = st.file_uploader("assets.csv", type=["csv"], key="assets_upload_v3")
        sbom_file = st.file_uploader("sbom.csv", type=["csv"], key="sbom_upload_v3")
    with col_u2:
        vulns_file = st.file_uploader("vulnerabilities.csv / cve_mapping.csv", type=["csv"], key="vulns_upload_v3")
        cti_file = st.file_uploader("cti.csv / mitre_mapping.csv", type=["csv"], key="cti_upload_v3")
    with col_u3:
        controls_file = st.file_uploader("controls.csv", type=["csv"], key="controls_upload_v3")
        labels_file = st.file_uploader("expert_labels.csv", type=["csv"], key="labels_upload_v3")
        business_file = st.file_uploader("business_impact.csv", type=["csv"], key="business_upload_v3")

    if st.button("🔄 Normalize uploaded real-data bundle", key="normalize_real_bundle"):
        raw_assets = read_csv_safely(assets_file)
        sbom_df = read_csv_safely(sbom_file)
        vulns_df = read_csv_safely(vulns_file)
        cti_df = read_csv_safely(cti_file)
        controls_df = read_csv_safely(controls_file)
        labels_df = read_csv_safely(labels_file)
        business_df = read_csv_safely(business_file)
        norm_assets = normalize_uploaded_assets(raw_assets, seed=rng_seed)
        norm_assets, norm_vulns = enrich_assets_with_vulnerabilities(norm_assets, vulns_df, sbom_df)
        st.session_state["real_data_bundle"] = {
            "assets_raw": raw_assets, "assets": norm_assets, "sbom": sbom_df,
            "vulnerabilities": norm_vulns, "cti": cti_df, "controls": controls_df,
            "expert_labels": labels_df, "business_impact": business_df,
        }
        st.success(f"Normalized bundle: {len(norm_assets)} assets, {len(norm_vulns)} vulnerability rows, {len(cti_df)} CTI rows.")

    bundle = st.session_state.get("real_data_bundle")
    if bundle:
        st.markdown("#### Normalized Asset Inventory")
        st.dataframe(bundle.get("assets", pd.DataFrame()).head(100), use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        assets_norm = bundle.get("assets", pd.DataFrame())
        vulns_norm = bundle.get("vulnerabilities", pd.DataFrame())
        c1.metric("Assets", len(assets_norm))
        c2.metric("Vulnerability rows", len(vulns_norm))
        c3.metric("Known exploited", int(vulns_norm.get("known_exploited", pd.Series(dtype=int)).sum()) if not vulns_norm.empty else 0)
        c4.metric("CTI rows", len(bundle.get("cti", pd.DataFrame())))

        if st.button("✅ Use uploaded assets as active environment", key="apply_real_env"):
            env_real = build_real_environment_from_uploads(assets_norm, seed=rng_seed)
            if env_real:
                st.session_state["env"] = env_real
                st.session_state["topology"] = env_real["topology_json"]
                st.success("Real-data environment is now active. You can run scenario generation / ML tabs using this asset base.")
            else:
                st.warning("No usable assets found. Please upload an assets.csv with at least asset_id/asset_type or hostname/type columns.")

        st.markdown("#### SBOM / CVE correlation summary")
        if not assets_norm.empty:
            summary_cols = [c for c in ["asset_id", "asset_type", "layer", "vuln_count", "cvss_weighted_avg_real", "epss_max_real", "known_exploited_count"] if c in assets_norm.columns]
            st.dataframe(assets_norm[summary_cols].sort_values("vuln_count", ascending=False).head(50), use_container_width=True)

        st.markdown("#### Probabilistic Attack Graph")
        active_env = st.session_state.get("env")
        if active_env is not None and st.button("🕸️ Compute top probabilistic attack paths", key="prob_paths"):
            prob_paths = compute_probabilistic_attack_paths(active_env["assets"], active_env["topology_json"], top_k=15)
            st.session_state["prob_paths"] = prob_paths
        if st.session_state.get("prob_paths") is not None:
            st.dataframe(st.session_state["prob_paths"], use_container_width=True)
    else:
        st.info("Upload files and click **Normalize uploaded real-data bundle**. The app also remains usable with synthetic/scalability data only.")

    st.markdown("#### Expected CSV columns")
    st.code("""assets.csv: asset_id,asset_type,zone,criticality,exposure,patch_compliance,control_coverage,asset_value
vulnerabilities.csv: asset_id,cve_id,cvss_score,epss_score,known_exploited
sbom.csv: asset_id,component_name,component_version,package_type,cve_id
cti.csv: threat_actor,mitre_technique,tactic,target_asset_type,confidence,source,first_seen,last_seen
expert_labels.csv: scenario_id,expert_risk_label,expert_risk_score""", language="text")

# ═══════════════════════════════════════════════════════════════════════════
# TAB: OPS + GOVERNANCE — MM-PASTA, FAIR, freshness, tickets, human review
# ═══════════════════════════════════════════════════════════════════════════
with tab_ops:
    st.subheader("🏛️ Continuous PASTA Ops + Governance")
    st.caption("Operationalizes the paper recommendations: MM-PASTA maturity, FAIR-style financial risk, model freshness/drift, risk-to-ticket export and human-AI review governance.")

    active_env = st.session_state.get("env")
    active_assets = active_env["assets"] if isinstance(active_env, dict) and "assets" in active_env else pd.DataFrame()

    ops_tabs = st.tabs(["MM-PASTA", "FAIR Risk", "Freshness/Drift", "Risk-to-Ticket", "Human Review"])

    with ops_tabs[0]:
        st.markdown("#### MM-PASTA Maturity Assessment")
        m1, m2, m3, m4 = st.columns(4)
        process_formalization = m1.slider("Process formalization", 0, 100, 45)
        tooling_integration = m2.slider("Tooling integration", 0, 100, 40)
        automation_depth = m3.slider("Automation depth", 0, 100, 35)
        scalability_outcome = m4.slider("Scalability outcome", 0, 100, 35)
        m5, m6, m7 = st.columns(3)
        model_freshness = m5.slider("Model freshness", 0, 100, 50)
        risk_ticket_conversion = m6.slider("Risk-to-ticket conversion", 0, 100, 30)
        coverage = m7.slider("Portfolio coverage", 0, 100, 40)
        maturity = assess_mm_pasta(process_formalization, tooling_integration, automation_depth, scalability_outcome, model_freshness, risk_ticket_conversion, coverage)
        st.session_state["maturity_results"] = maturity
        c1, c2 = st.columns(2)
        c1.metric("MM-PASTA Score", maturity["score"])
        c2.metric("Maturity Level", f"Level {maturity['level']} — {maturity['level_name']}")
        if maturity["recommendations"]:
            st.markdown("**Recommended next steps**")
            for rec in maturity["recommendations"]:
                st.write(f"- {rec}")

    with ops_tabs[1]:
        st.markdown("#### FAIR-style Financial Risk Quantification")
        f1, f2 = st.columns(2)
        asset_value_default = f1.number_input("Default asset value", min_value=0.0, value=100000.0, step=10000.0)
        control_cost_default = f2.number_input("Default control cost", min_value=0.0, value=15000.0, step=1000.0)
        if st.button("💰 Calculate FAIR-style exposure", key="fair_calc"):
            fair_df = compute_fair_results(active_assets, st.session_state.get("features"), asset_value_default, control_cost_default)
            st.session_state["fair_results"] = fair_df
        if st.session_state.get("fair_results") is not None:
            fair_df = st.session_state["fair_results"]
            st.dataframe(fair_df.head(100), use_container_width=True)
            if not fair_df.empty:
                st.metric("Total annualized loss expectancy", f"{fair_df['annualized_loss_expectancy'].sum():,.0f}")

    with ops_tabs[2]:
        st.markdown("#### Model Freshness and Architecture Drift")
        last_update = st.date_input("Last threat-model update date", value=datetime.now().date())
        prev_assets_file = st.file_uploader("Optional previous_assets.csv", type=["csv"], key="prev_assets_v3")
        prev_vulns_file = st.file_uploader("Optional previous_vulnerabilities.csv", type=["csv"], key="prev_vulns_v3")
        if st.button("🧭 Calculate freshness/drift", key="freshness_calc"):
            prev_assets = normalize_uploaded_assets(read_csv_safely(prev_assets_file), seed=rng_seed)
            prev_vulns = read_csv_safely(prev_vulns_file)
            current_vulns = (st.session_state.get("real_data_bundle") or {}).get("vulnerabilities", pd.DataFrame())
            freshness = compute_freshness_and_drift(active_assets, prev_assets, last_update, current_vulns, prev_vulns)
            st.session_state["freshness_results"] = freshness
        if st.session_state.get("freshness_results"):
            fr = st.session_state["freshness_results"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Freshness score", fr["model_freshness_score"])
            c2.metric("Days stale", fr["days_since_last_update"])
            c3.metric("New assets", fr["new_assets"])
            c4.metric("New CVEs", fr["new_cves"])
            st.warning("Reassessment recommended") if fr["reassessment_recommended"] else st.success("No immediate reassessment trigger detected")

    with ops_tabs[3]:
        st.markdown("#### Risk-to-Ticket Backlog Export")
        mit_df = st.session_state.get("mitigation_results", pd.DataFrame())
        fair_df = st.session_state.get("fair_results", pd.DataFrame())
        if st.button("🎫 Generate remediation backlog", key="ticket_gen"):
            tickets = create_ticket_backlog(mit_df, fair_df, st.session_state.get("features"))
            st.session_state["ticket_backlog"] = tickets
        if st.session_state.get("ticket_backlog") is not None:
            tickets = st.session_state["ticket_backlog"]
            st.dataframe(tickets, use_container_width=True)
            st.download_button("📥 Jira/GitHub/ServiceNow CSV", tickets.to_csv(index=False).encode(), "pasta_risk_backlog.csv", "text/csv")
            st.download_button("📥 ServiceNow-style JSON", tickets.to_json(orient="records", indent=2).encode(), "pasta_risk_backlog.json", "application/json")

    with ops_tabs[4]:
        st.markdown("#### Human-AI Governance Review")
        scenarios = st.session_state.get("scenarios", pd.DataFrame())
        if scenarios is not None and not scenarios.empty:
            review_sample = scenarios.head(50).copy()
            if "scenario_id" not in review_sample.columns:
                review_sample.insert(0, "scenario_id", [f"S-{i+1:04d}" for i in range(len(review_sample))])
            review_sample["human_reviewed"] = False
            review_sample["review_decision"] = "Pending"
            review_sample["reviewer_comment"] = ""
            edited = st.data_editor(review_sample, use_container_width=True, num_rows="dynamic", key="review_editor_v3")
            st.session_state["review_log"] = edited
            st.download_button("📥 Human review log", edited.to_csv(index=False).encode(), "human_ai_review_log.csv", "text/csv")
        else:
            st.info("Generate scenarios first to create a review log.")

    st.divider()
    st.markdown("#### PASTA Interchange Format Preview")
    if st.button("🧾 Build PIF JSON", key="build_pif"):
        config = {"n_assets": n_assets, "seed": rng_seed, "n_scenarios": n_scenarios, "vectors": selected_vecs}
        st.session_state["pif_bundle"] = build_pif_export(st.session_state, config)
    if st.session_state.get("pif_bundle"):
        st.download_button("📥 Download PASTA Interchange Format JSON", json.dumps(st.session_state["pif_bundle"], indent=2).encode(), "pasta_interchange_format.json", "application/json")
        st.json({k: ("..." if isinstance(v, dict) else v) for k, v in st.session_state["pif_bundle"].items()})


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

        if st.session_state["mc_events"] is not None:
            st.download_button("📥 Monte-Carlo Event Dataset (CSV)",
                st.session_state["mc_events"].to_csv(index=False).encode(),
                "mc_event_dataset.csv", "text/csv")

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

        if st.session_state["clf_results"] is not None and "error" not in st.session_state["clf_results"]:
            clf_res = st.session_state["clf_results"]
            clf_rows = []
            for name, r in clf_res.items():
                clf_rows.append({
                    "Model":     name,
                    "accuracy":  r["accuracy"], "precision": r["precision"],
                    "recall":    r["recall"],   "f1":        r["f1"],
                    "roc_auc":   r["roc_auc"],  "pr_auc":    r["pr_auc"],
                    "cv_f1_mean":r["cv_f1_mean"], "cv_f1_std": r["cv_f1_std"],
                    "train_time_s": r["train_time_s"], "infer_ms": r["infer_ms"],
                    "n_train": r["n_train"], "n_test": r["n_test"],
                })
            st.download_button("📥 Alerting Classifier Metrics (CSV)",
                pd.DataFrame(clf_rows).to_csv(index=False).encode(),
                "alerting_classifier_metrics.csv", "text/csv")

        # Full config JSON
        config = {
            "n_assets": n_assets, "seed": rng_seed,
            "asset_mix": asset_mix, "threat_actors": selected_actors,
            "n_scenarios": n_scenarios, "attack_vectors": selected_vecs,
            "max_path_len": max_path_len, "test_size": test_size,
            "cv_folds": cv_folds, "rf_params": rf_params, "gb_params": gb_params,
            "bench_sizes": list(bench_sizes),
            # NEW — Step 5b parameters
            "mc_n_sims":     mc_n_sims,
            "mc_steps":      mc_steps,
            "mc_epsilon":    mc_epsilon,
            "mc_norm_alert": mc_norm_alert,
        }
        st.download_button("🧾 Full Experiment Config (JSON)",
            json.dumps(config, indent=2).encode(), "experiment_config.json", "application/json")

        if st.session_state.get("ticket_backlog") is not None:
            st.download_button("📥 Risk-to-Ticket Backlog (CSV)",
                st.session_state["ticket_backlog"].to_csv(index=False).encode(),
                "pasta_risk_backlog.csv", "text/csv")

        if st.session_state.get("fair_results") is not None:
            st.download_button("📥 FAIR Financial Risk Results (CSV)",
                st.session_state["fair_results"].to_csv(index=False).encode(),
                "fair_financial_risk.csv", "text/csv")

        if st.session_state.get("pif_bundle") is not None:
            st.download_button("📥 PASTA Interchange Format (JSON)",
                json.dumps(st.session_state["pif_bundle"], indent=2).encode(),
                "pasta_interchange_format.json", "application/json")


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
