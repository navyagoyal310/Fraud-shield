import os
import re
import math
import io
import time
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & ANIMATED FUTURISTIC SAAS STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BrainShield AI — Dynamic Threat Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_cyber_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #030712;
        color: #F3F4F6;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1400px;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* Background Animations */
    .cyber-bg {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        z-index: -2;
        background: radial-gradient(circle at 50% 20%, #0F172A 0%, #030712 85%);
    }

    .cyber-grid {
        position: fixed;
        top: 0; left: 0;
        width: 200%; height: 200%;
        z-index: -1;
        background-image: 
            linear-gradient(to right, rgba(56, 189, 248, 0.03) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(168, 85, 247, 0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        animation: gridMove 25s linear infinite;
        pointer-events: none;
    }

    @keyframes gridMove {
        0% { transform: translate(0, 0); }
        100% { transform: translate(-40px, -40px); }
    }

    @keyframes pulseGlow {
        0% { filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.4)); }
        50% { filter: drop-shadow(0 0 25px rgba(168, 85, 247, 0.7)); }
        100% { filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.4)); }
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(168, 85, 247, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(168, 85, 247, 0.12);
    }

    /* Brand Header */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 10px;
    }

    .brand-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        background: linear-gradient(135deg, #06B6D4 0%, #38BDF8 40%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    .hero-tagline {
        font-size: 1.2rem;
        font-weight: 600;
        color: #38BDF8;
        letter-spacing: 0.5px;
        margin-bottom: 20px;
    }

    /* Status Engine Card */
    .status-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.8));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 16px 22px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #E2E8F0;
    }

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
        animation: blink 1.8s infinite ease-in-out;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Dynamic Scoreboard HUD */
    .hud-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .hud-card:hover { transform: scale(1.02); }
    .hud-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .hud-lbl {
        font-size: 0.75rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Timeline Stepper */
    .timeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 15px;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.1);
        overflow-x: auto;
    }

    .timeline-step {
        text-align: center;
        position: relative;
        flex: 1;
    }

    .timeline-step::after {
        content: '→';
        position: absolute;
        top: 20%;
        right: -10px;
        color: #A855F7;
        font-size: 1.2rem;
    }

    .timeline-step:last-child::after { content: ''; }

    .step-node {
        width: 36px; height: 36px;
        border-radius: 50%;
        background: #0F172A;
        border: 2px solid #38BDF8;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 6px auto;
        font-weight: bold; font-size: 0.85rem; color: #38BDF8;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }

    .step-text { font-size: 0.75rem; color: #CBD5E1; font-weight: 600; }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        color: #9CA3AF;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 0px 20px;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #06B6D4 0%, #38BDF8 50%, #A855F7 100%);
        color: #030712;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.95rem;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.35);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(168, 85, 247, 0.5);
    }

    /* Responsive Grid Tweaks */
    @media (max-width: 768px) {
        .brand-title { font-size: 2.2rem; }
        .timeline-container { flex-direction: column; gap: 12px; }
        .timeline-step::after { content: '↓'; top: 100%; right: 45%; }
    }
    </style>
    <div class="cyber-bg"></div>
    <div class="cyber-grid"></div>
    """, unsafe_allow_html=True)

inject_cyber_styles()

# -----------------------------------------------------------------------------
# 2. FEATURE EXTRACTION & NORMALIZATION ENGINE
# -----------------------------------------------------------------------------
def normalize_and_validate_url(url_str: str) -> tuple[bool, str, str]:
    if not url_str or not isinstance(url_str, str):
        return False, "", "Empty URL string provided."
    clean_url = url_str.strip().lower()
    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        clean_url = "http://" + clean_url
    try:
        parsed = urlparse(clean_url)
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
        ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        if not domain or (not ip_pattern.match(domain.split(':')[0]) and '.' not in domain):
            return False, clean_url, "Invalid URL structural format."
        return True, clean_url, "Valid URL"
    except Exception as e:
        return False, clean_url, "Malformed URL: " + str(e)

FEATURE_NAMES = [
    'url_length', 'domain_length', 'num_dots', 'num_hyphens', 'num_underline',
    'num_slash', 'num_digits', 'num_letters', 'has_https', 'is_ip',
    'num_subdomains', 'suspicious_keywords_count', 'special_char_count',
    'entropy', 'has_at_symbol', 'has_port'
]

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'update', 'account', 'banking', 'secure', 'paypal',
    'free', 'crypto', 'bonus', 'claim', 'winner', 'confirm', 'support',
    'wallet', 'signin', 'auth', 'gift', 'prize', 'tokn', 'pass', 'checkout'
]

def calculate_entropy(url_str: str) -> float:
    if not url_str:
        return 0.0
    entropy = 0.0
    for x in set(url_str):
        p_x = float(url_str.count(x)) / len(url_str)
        entropy -= p_x * math.log2(p_x)
    return float(entropy)

def extract_features(raw_url: str) -> dict:
    _, url, _ = normalize_and_validate_url(raw_url)
    parsed = urlparse(url)
    domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
    clean_domain = domain.split(':')[0]
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    is_ip = 1 if ip_pattern.match(clean_domain) else 0
    subdomains = clean_domain.split('.')
    num_subdomains = max(0, len(subdomains) - 2) if not is_ip else 0

    return {
        'url_length': len(url),
        'domain_length': len(domain),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underline': url.count('_'),
        'num_slash': url.count('/'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_letters': sum(c.isalpha() for c in url),
        'has_https': 1 if url.startswith('https://') else 0,
        'is_ip': is_ip,
        'num_subdomains': num_subdomains,
        'suspicious_keywords_count': sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url),
        'special_char_count': len(re.findall(r'[@%&=?+~#$!]', url)),
        'entropy': round(calculate_entropy(url), 4),
        'has_at_symbol': 1 if '@' in url else 0,
        'has_port': 1 if len(domain.split(':')) > 1 else 0
    }

# -----------------------------------------------------------------------------
# 3. MACHINE LEARNING & EXPLAINABILITY ENGINE
# -----------------------------------------------------------------------------
MODEL_FILE = "brainshield_rf_10k_model.joblib"

def generate_large_synthetic_dataset(n_samples: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    data = []
    half = n_samples // 2
    
    # Safe vectors
    for _ in range(half):
        data.append([
            int(np.random.normal(28, 6)), int(np.random.normal(12, 3)), 1, 0, 0, 2, 
            int(np.random.poisson(1)), 20, 1, 0, 0, 0, 0, 
            float(np.random.normal(3.7, 0.3)), 0, 0, 0
        ])
    # Fraud vectors
    for _ in range(half):
        data.append([
            int(np.random.normal(82, 18)), int(np.random.normal(28, 7)), 3, 2, 1, 5, 
            int(np.random.normal(14, 4)), 38, 0, 0, 2, 2, 4, 
            float(np.random.normal(4.9, 0.4)), 0, 0, 1
        ])
    cols = FEATURE_NAMES + ['is_fraud']
    return pd.DataFrame(data, columns=cols).clip(lower=0)

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass
    df = generate_large_synthetic_dataset(10000)
    X, y = df[FEATURE_NAMES], df['is_fraud']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    return model

model = load_or_train_model()

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

def classify_threat_category(url: str, features: dict, fraud_prob: float) -> str:
    if fraud_prob < 0.35:
        return "Legitimate / Enterprise Verified"
    url_lower = url.lower()
    if any(k in url_lower for k in ['paypal', 'bank', 'secure', 'login', 'auth', 'verify', 'account']):
        return "Credential Theft / Active Phishing"
    elif any(k in url_lower for k in ['shop', 'store', 'cart', 'discount', 'checkout']):
        return "Fraudulent E-Commerce / Scam Store"
    elif any(k in url_lower for k in ['crypto', 'wallet', 'binance', 'btc', 'claim', 'tokn']):
        return "Crypto Drainer / Token Scam"
    elif features['is_ip'] == 1 or features['entropy'] > 4.8:
        return "Malware Host / Exploit Node"
    else:
        return "Suspicious Domain / Cyber Risk"

def generate_category_recommendations(category: str, features: dict) -> tuple[list, list, list]:
    risk_factors = []
    safe_indicators = []
    recs = []

    # Safe indicators
    if features['has_https'] == 1:
        safe_indicators.append("Encrypted SSL/TLS communications protocol active (HTTPS).")
    if features['is_ip'] == 0:
        safe_indicators.append("Standard domain name resolution active (non-IP addressing).")
    if features['url_length'] < 45:
        safe_indicators.append("Concise, non-obfuscated URL structural length.")

    # Risk factors
    if features['has_https'] == 0:
        risk_factors.append("Unencrypted connection protocol (HTTP detected).")
    if features['is_ip'] == 1:
        risk_factors.append("Domain routes directly to a raw public IP address.")
    if features['url_length'] > 65:
        risk_factors.append(f"Excessive string length ({features['url_length']} characters).")
    if features['suspicious_keywords_count'] > 0:
        risk_factors.append(f"Detected {features['suspicious_keywords_count']} high-risk target phishing keywords.")
    if features['entropy'] > 4.6:
        risk_factors.append(f"High string randomness/entropy ({features['entropy']}).")

    # Category-Specific Actions
    if category == "Credential Theft / Active Phishing":
        recs.append("Enforce strict OAuth token verification and MFA hardware keys.")
        recs.append("Block domain across enterprise DNS and email gateways.")
    elif category == "Crypto Drainer / Token Scam":
        recs.append("Block Web3 wallet contract signature requests on unverified origin.")
        recs.append("Flag associated wallet approval addresses to compliance databases.")
    elif category == "Fraudulent E-Commerce / Scam Store":
        recs.append("Verify payment gateway merchant ID and WHOIS domain registration age.")
        recs.append("Warn users before entering credit card or payment credentials.")
    elif category == "Malware Host / Exploit Node":
        recs.append("Isolate network node and run automated endpoint virus scan.")
        recs.append("Block inbound/outbound TCP traffic to the target IP address.")
    else:
        recs.append("Exercise baseline caution; avoid entering confidential organization data.")

    if not risk_factors:
        risk_factors.append("Structural parameters conform to standard safe baseline distributions.")
    if not safe_indicators:
        safe_indicators.append("Limited baseline safety indicators found.")

    return risk_factors, safe_indicators, recs

def compute_shap_explanations(feat_df: pd.DataFrame) -> pd.DataFrame:
    try:
        explainer = get_shap_explainer(model)
        shap_vals = explainer.shap_values(feat_df)
        vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        return pd.DataFrame({'Feature': FEATURE_NAMES, 'SHAP_Impact': vals, 'Value': feat_df.iloc[0].values}).sort_values(by='SHAP_Impact', ascending=False)
    except Exception:
        return pd.DataFrame({'Feature': FEATURE_NAMES, 'SHAP_Impact': model.feature_importances_, 'Value': feat_df.iloc[0].values}).sort_values(by='SHAP_Impact', ascending=False)

def analyze_single_url(url: str) -> dict:
    feats = extract_features(url)
    feat_df = pd.DataFrame([feats])[FEATURE_NAMES]
    fraud_prob = float(model.predict_proba(feat_df)[0][1])
    trust_score = max(0, min(100, int((1.0 - fraud_prob) * 100)))
    confidence_score = round(max(fraud_prob, 1.0 - fraud_prob) * 100, 1)
    
    threat_level = "Safe" if fraud_prob < 0.20 else ("Low Risk" if fraud_prob < 0.40 else ("Medium Risk" if fraud_prob < 0.65 else ("High Risk" if fraud_prob < 0.85 else "Critical Risk")))
    threat_category = classify_threat_category(url, feats, fraud_prob)
    shap_df = compute_shap_explanations(feat_df)
    risk_factors, safe_indicators, recs = generate_category_recommendations(threat_category, feats)

    return {
        'url': url, 'fraud_probability': fraud_prob, 'trust_score': trust_score,
        'confidence_score': confidence_score, 'threat_level': threat_level, 
        'threat_category': threat_category, 'features': feats, 
        'shap_importance': shap_df, 'risk_factors': risk_factors, 
        'safe_indicators': safe_indicators, 'recommendations': recs
    }

# -----------------------------------------------------------------------------
# 4. PLOTLY CIRCULAR GAUGE & AUDIT REPORTING
# -----------------------------------------------------------------------------
def create_circular_trust_gauge(score: float, title: str, is_trust: bool = True) -> go.Figure:
    if is_trust:
        color = "#38BDF8" if score >= 70 else ("#F59E0B" if score >= 40 else "#EF4444")
        suffix = "/100"
    else:
        color = "#EF4444" if score >= 60 else ("#F59E0B" if score >= 30 else "#10B981")
        suffix = "%"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': suffix, 'font': {'color': color, 'size': 38, 'family': 'Syne'}},
        title={'text': title, 'font': {'size': 14, 'color': '#9CA3AF', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "rgba(15, 23, 42, 0.4)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(56, 189, 248, 0.1)'}
            ],
        }
    ))
    fig.update_layout(height=210, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def generate_pdf_report(analysis: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = Paragraph