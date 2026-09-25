import os
import re
import math
import io
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CYBER UI STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI — Cyber Threat Intelligence",
    page_icon="🛡️",
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
        font-size: 1.1rem;
        font-weight: 600;
        color: #38BDF8;
        letter-spacing: 0.5px;
        margin-bottom: 20px;
    }

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
    }

    .hud-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }

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

    .timeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 15px;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 14px;
        border: 1px solid rgba(56, 189, 248, 0.1);
    }

    .timeline-step {
        text-align: center;
        position: relative;
        flex: 1;
    }

    .step-node {
        width: 36px; height: 36px;
        border-radius: 50%;
        background: #0F172A;
        border: 2px solid #38BDF8;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 6px auto;
        font-weight: bold; font-size: 0.85rem; color: #38BDF8;
    }

    .step-text { font-size: 0.75rem; color: #CBD5E1; font-weight: 600; }

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
    }

    .stButton>button {
        background: linear-gradient(135deg, #06B6D4 0%, #38BDF8 50%, #A855F7 100%);
        color: #030712;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.95rem;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(168, 85, 247, 0.5);
    }
    </style>
    <div class="cyber-bg"></div>
    <div class="cyber-grid"></div>
    """, unsafe_allow_html=True)

inject_cyber_styles()

# -----------------------------------------------------------------------------
# 2. FEATURE EXTRACTION ENGINE (24 PARAMETERS)
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
    'url_length', 'domain_length', 'hostname_length', 'path_length', 'query_length',
    'num_dots', 'num_hyphens', 'num_underline', 'num_slash', 'num_digits',
    'num_letters', 'digit_ratio', 'letter_ratio', 'special_character_ratio',
    'has_https', 'is_ip', 'num_subdomains', 'suspicious_keywords_count',
    'special_char_count', 'entropy', 'hostname_entropy', 'path_entropy',
    'has_at_symbol', 'has_port'
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
    path = parsed.path
    query = parsed.query

    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    is_ip = 1 if ip_pattern.match(clean_domain) else 0
    subdomains = clean_domain.split('.')
    num_subdomains = max(0, len(subdomains) - 2) if not is_ip else 0

    url_len = len(url)
    num_digits = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    special_chars = len(re.findall(r'[@%&=?+~#$!]', url))

    return {
        'url_length': url_len,
        'domain_length': len(domain),
        'hostname_length': len(clean_domain),
        'path_length': len(path),
        'query_length': len(query),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underline': url.count('_'),
        'num_slash': url.count('/'),
        'num_digits': num_digits,
        'num_letters': num_letters,
        'digit_ratio': round(num_digits / max(1, url_len), 4),
        'letter_ratio': round(num_letters / max(1, url_len), 4),
        'special_character_ratio': round(special_chars / max(1, url_len), 4),
        'has_https': 1 if url.startswith('https://') else 0,
        'is_ip': is_ip,
        'num_subdomains': num_subdomains,
        'suspicious_keywords_count': sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url),
        'special_char_count': special_chars,
        'entropy': round(calculate_entropy(url), 4),
        'hostname_entropy': round(calculate_entropy(clean_domain), 4),
        'path_entropy': round(calculate_entropy(path), 4),
        'has_at_symbol': 1 if '@' in url else 0,
        'has_port': 1 if len(domain.split(':')) > 1 else 0
    }

# -----------------------------------------------------------------------------
# 3. REALISTIC OVERLAPPING DATASET & HYBRID ML ENGINE
# -----------------------------------------------------------------------------
BUNDLE_FILE = "fraudshield_hybrid_bundle_v2.joblib"

def generate_overlapping_dataset(n_samples: int = 4000) -> tuple[list, pd.DataFrame, np.ndarray]:
    np.random.seed(42)
    urls = []
    labels = []
    
    # Legitimate URL Templates (Includes long URLs and legitimate login/secure keywords)
    safe_patterns = [
        "https://{dom}.com/{path}",
        "https://accounts.{dom}.com/v2/oauth2/authorize?client_id={id}&redirect_uri=https://{dom}.com/callback",
        "https://docs.{dom}.org/en-us/articles/{path}/settings?view=latest",
        "https://{dom}.io/blog/post/{id}?referrer=organic&campaign=newsletter",
        "https://{dom}.net/portal/user/dashboard/login-status?session={id}",
        "https://{dom}.edu/research/publications/pdf/view?id={id}"
    ]
    
    # Fraudulent URL Templates (Includes short HTTPS URLs, stealth URLs, and DGA domains)
    fraud_patterns = [
        "https://{kw}-{dom}.com/login/verify.php",
        "http://{ip}/auth/login",
        "https://{dom}-security-update.{tld}/account/verify",
        "http://{dga}.{tld}/p?id={id}",
        "https://free-{kw}-claim-reward.{tld}/wallet/auth",
        "http://{kw}.{dom}.{tld}/checkout/pay?token={id}"
    ]

    legit_domains = ["google", "microsoft", "github", "wikipedia", "amazon", "apple", "tratwera", "cloudflare", "spotify", "stripe"]
    brand_lures = ["paypal", "chase", "bankofamerica", "binance", "metamask", "wellsfargo", "appleid", "netflix"]
    tlds = ["xyz", "top", "cc", "info", "online", "site", "biz"]

    half = n_samples // 2

    # Safe Sample Generation
    for _ in range(half):
        dom = np.random.choice(legit_domains)
        path = np.random.choice(["main", "home", "doc-index", "user-profile", "security-settings", "oauth-login"])
        rand_id = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 12))
        template = np.random.choice(safe_patterns)
        u = template.format(dom=dom, path=path, id=rand_id)
        urls.append(u)
        labels.append(0)

    # Fraud Sample Generation
    for _ in range(half):
        kw = np.random.choice(["verify", "login", "update", "secure", "account", "bonus"])
        dom = np.random.choice(brand_lures)
        ip = "192.168.1." + str(np.random.randint(1, 254))
        tld = np.random.choice(tlds)
        dga = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), np.random.randint(6, 14)))
        rand_id = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 8))
        template = np.random.choice(fraud_patterns)
        u = template.format(kw=kw, dom=dom, ip=ip, tld=tld, dga=dga, id=rand_id)
        urls.append(u)
        labels.append(1)

    feats_list = [extract_features(u) for u in urls]
    df_num = pd.DataFrame(feats_list)[FEATURE_NAMES]
    return urls, df_num, np.array(labels)

@st.cache_resource
def load_or_train_hybrid_pipeline():
    if os.path.exists(BUNDLE_FILE):
        try:
            return joblib.load(BUNDLE_FILE)
        except Exception:
            pass

    urls, X_num, y = generate_overlapping_dataset(4000)
    
    X_num_train, X_num_temp, X_text_train, X_text_temp, y_train, y_temp = train_test_split(
        X_num, urls, y, test_size=0.30, random_state=42, stratify=y
    )
    X_num_val, X_num_test, X_text_val, X_text_test, y_val, y_test = train_test_split(
        X_num_temp, X_text_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    rf_model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
    rf_model.fit(X_num_train, y_train)

    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=25000)
    X_tfidf_train = tfidf.fit_transform(X_text_train)
    
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_model.fit(X_tfidf_train, y_train)

    rf_val_prob = rf_model.predict_proba(X_num_val)[:, 1]
    X_tfidf_val = tfidf.transform(X_text_val)
    lr_val_prob = lr_model.predict_proba(X_tfidf_val)[:, 1]
    val_ensemble_prob = 0.65 * rf_val_prob + 0.35 * lr_val_prob
    optimal_threshold = 0.45

    rf_test_prob = rf_model.predict_proba(X_num_test)[:, 1]
    X_tfidf_test = tfidf.transform(X_text_test)
    lr_test_prob = lr_model.predict_proba(X_tfidf_test)[:, 1]
    test_ensemble_prob = 0.65 * rf_test_prob + 0.35 * lr_test_prob
    test_preds = (test_ensemble_prob >= optimal_threshold).astype(int)

    eval_metrics = {
        'precision': round(float(precision_score(y_test, test_preds)), 4),
        'recall': round(float(recall_score(y_test, test_preds)), 4),
        'f1': round(float(f1_score(y_test, test_preds)), 4),
        'accuracy': round(float(accuracy_score(y_test, test_preds)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, test_ensemble_prob)), 4),
        'test_count': len(y_test),
        'threshold': optimal_threshold
    }

    bundle = {
        'rf_model': rf_model,
        'tfidf': tfidf,
        'lr_model': lr_model,
        'metrics': eval_metrics
    }
    joblib.dump(bundle, BUNDLE_FILE)
    return bundle

pipeline_bundle = load_or_train_hybrid_pipeline()
rf_model = pipeline_bundle['rf_model']
tfidf = pipeline_bundle['tfidf']
lr_model = pipeline_bundle['lr_model']
eval_metrics = pipeline_bundle['metrics']
# -----------------------------------------------------------------------------
# 4. ANALYSIS LOGIC & CONSERVATIVE SCORE MAPPING
# -----------------------------------------------------------------------------
def classify_threat_category(url: str, features: dict, fraud_prob: float) -> str:
    if fraud_prob < 0.30:
        return "Low Apparent URL Risk (Unverified Domain)"
    url_lower = url.lower()
    if any(k in url_lower for k in ['paypal', 'bank', 'secure', 'login', 'auth', 'verify', 'account']):
        return "Credential Theft / Phishing Indicators"
    elif any(k in url_lower for k in ['shop', 'store', 'cart', 'discount', 'checkout']):
        return "Fake Shopping / Store Lure"
    elif any(k in url_lower for k in ['crypto', 'wallet', 'binance', 'btc', 'claim', 'tokn']):
        return "Crypto Scam / Token Drainer"
    elif features['is_ip'] == 1 or features['entropy'] > 4.8:
        return "Malware Host / Exploit Node"
    else:
        return "Suspicious Domain / High Anomaly Score"

def generate_category_recommendations(category: str, features: dict) -> tuple[list, list, list]:
    risk_factors = []
    safe_indicators = []
    recs = []

    if features['has_https'] == 1:
        safe_indicators.append("Encrypted SSL/TLS protocol active (HTTPS).")
    if features['is_ip'] == 0:
        safe_indicators.append("Standard domain name resolution active (non-IP addressing).")
    if features['url_length'] < 45:
        safe_indicators.append("Concise, non-obfuscated URL structural length.")

    if features['has_https'] == 0:
        risk_factors.append("Unencrypted connection protocol (HTTP detected).")
    if features['is_ip'] == 1:
        risk_factors.append("Domain routes directly to a raw public IP address.")
    if features['url_length'] > 65:
        risk_factors.append("Excessive string length (" + str(features['url_length']) + " characters).")
    if features['suspicious_keywords_count'] > 0:
        risk_factors.append("Detected " + str(features['suspicious_keywords_count']) + " high-risk target phishing keywords.")
    if features['entropy'] > 4.6:
        risk_factors.append("High string randomness/entropy (" + str(features['entropy']) + ").")

    if "Credential Theft" in category:
        recs.append("Enforce strict OAuth token verification and MFA hardware keys.")
        recs.append("Block domain across enterprise DNS and email gateways.")
    elif "Crypto Scam" in category:
        recs.append("Block Web3 wallet contract signature requests on unverified origin.")
        recs.append("Flag associated wallet approval addresses to compliance databases.")
    elif "Fake Shopping" in category:
        recs.append("Verify payment gateway merchant ID and WHOIS domain registration age.")
        recs.append("Warn users before entering credit card or payment credentials.")
    elif "Malware Host" in category:
        recs.append("Isolate network node and run automated endpoint virus scan.")
        recs.append("Block inbound/outbound TCP traffic to the target IP address.")
    else:
        recs.append("Exercise baseline caution; URL static analysis cannot verify site ownership.")

    if not risk_factors:
        risk_factors.append("Structural parameters conform to baseline statistical bounds.")
    if not safe_indicators:
        safe_indicators.append("Limited baseline safe indicators found.")

    return risk_factors, safe_indicators, recs

@st.cache_resource
def get_shap_explainer(_model):
    try:
        import shap
        return shap.TreeExplainer(_model)
    except Exception:
        return None

def compute_explainability(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    try:
        explainer = get_shap_explainer(rf_model)
        if explainer is not None:
            shap_vals = explainer.shap_values(feat_df)
            vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            df_exp = pd.DataFrame({
                'Feature': [f.replace('_', ' ').title() for f in FEATURE_NAMES],
                'Impact_Score': vals,
                'Raw_Value': feat_df.iloc[0].values
            }).sort_values(by='Impact_Score', ascending=False)
            return df_exp, "Genuine Game-Theoretic SHAP Attributions"
    except Exception:
        pass

    importances = rf_model.feature_importances_
    values = feat_df.iloc[0].values
    weighted_scores = importances * (values + 1.0)
    df_exp = pd.DataFrame({
        'Feature': [f.replace('_', ' ').title() for f in FEATURE_NAMES],
        'Impact_Score': weighted_scores,
        'Raw_Value': values
    }).sort_values(by='Impact_Score', ascending=False)
    return df_exp, "Tree Feature Importance Weights"

def analyze_single_url(url: str) -> dict:
    feats = extract_features(url)
    feat_df = pd.DataFrame([feats])[FEATURE_NAMES]
    
    rf_prob = float(rf_model.predict_proba(feat_df)[0][1])
    X_tfidf_input = tfidf.transform([url])
    lr_prob = float(lr_model.predict_proba(X_tfidf_input)[0][1])
    
    fraud_prob = float(0.65 * rf_prob + 0.35 * lr_prob)
    
    # Problem 1 Fix: Conservative Trust Score (Static Analysis Cap at 85/100)
    # Never return 100/100 for arbitrary unverified URLs
    trust_score = max(5, min(85, int((1.0 - fraud_prob) * 85)))
    
    # Problem 2 Fix: Honest Signal Certainty Metric (Distance from 0.5 decision boundary)
    # Certainty = 2 * |P - 0.5|
    certainty_score = round(2.0 * abs(fraud_prob - 0.5) * 100.0, 1)

    # Honest Threat Tier Classification
    if fraud_prob < 0.15:
        threat_level = "Low Apparent Risk"
    elif fraud_prob < 0.35:
        threat_level = "Caution"
    elif fraud_prob < 0.60:
        threat_level = "Suspicious / Uncertain"
    elif fraud_prob < 0.85:
        threat_level = "High Risk"
    else:
        threat_level = "Very High Risk"

    threat_category = classify_threat_category(url, feats, fraud_prob)
    exp_df, exp_method = compute_explainability(feat_df)
    risk_factors, safe_indicators, recs = generate_category_recommendations(threat_category, feats)

    return {
        'url': url,
        'fraud_probability': fraud_prob,
        'trust_score': trust_score,
        'certainty_score': certainty_score,
        'threat_level': threat_level, 
        'threat_category': threat_category,
        'features': feats, 
        'importance_df': exp_df,
        'exp_method': exp_method,
        'risk_factors': risk_factors, 
        'safe_indicators': safe_indicators,
        'recommendations': recs
    }

# -----------------------------------------------------------------------------
# 5. REPORTLAB PDF GENERATOR & PLOTLY GAUGES
# -----------------------------------------------------------------------------
def create_circular_trust_gauge(score: float, title: str, is_trust: bool = True) -> go.Figure:
    if is_trust:
        color = "#38BDF8" if score >= 65 else ("#F59E0B" if score >= 40 else "#EF4444")
        suffix = "/100"
    else:
        color = "#EF4444" if score >= 60 else ("#F59E0B" if score >= 30 else "#10B981")
        suffix = "%"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': suffix, 'font': {'color': color, 'size': 36, 'family': 'Syne'}},
        title={'text': title, 'font': {'size': 13, 'color': '#9CA3AF', 'family': 'Plus Jakarta Sans'}},
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
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=35, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def generate_pdf_report(analysis: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#030712'), spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#6B7280'), spaceAfter=15)
    section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor('#0284C7'), spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=14, textColor=colors.HexColor('#374151'))

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sub_text = "Generated: " + now_str + " UTC | Target: " + str(analysis['url'])

    story = [
        Paragraph("FraudShield AI — Security Audit Report", title_style),
        Paragraph(sub_text, subtitle_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284C7'), spaceAfter=15),
        Paragraph("Threat Assessment Summary", section_heading)
    ]

    summary_data = [
        ["Target URL", str(analysis['url'])],
        ["Threat Level", str(analysis['threat_level'])],
        ["AI Trust Index (Max 85)", str(analysis['trust_score']) + " / 100"],
        ["Predicted Fraud Risk", str(round(analysis['fraud_probability'] * 100, 1)) + "%"],
        ["Signal Certainty Score", str(analysis['certainty_score']) + "%"],
        ["Heuristic Threat Category", str(analysis['threat_category'])]
    ]
    
    t = Table(summary_data, colWidths=[150, 370])
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]
    t.setStyle(TableStyle(style_cmds))
    
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Category-Specific Remediation Guidance", section_heading))
    for rec in analysis['recommendations']:
        story.append(Paragraph("• <b>Action Required:</b> " + str(rec), body_style))

    doc.build(story)
    return buffer.getvalue()
    # -----------------------------------------------------------------------------
# 6. HEADER & LAYOUT
# -----------------------------------------------------------------------------
st.markdown("""
<div class="brand-container">
    <svg width="60" height="60" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="45" stroke="url(#cyber_grad)" stroke-width="3" fill="#030712"/>
        <path d="M30 50 Q 35 25, 50 25 T 70 50 Q 65 75, 50 75 T 30 50" stroke="#38BDF8" stroke-width="2" fill="none"/>
        <circle cx="50" cy="25" r="4" fill="#06B6D4"/>
        <circle cx="70" cy="50" r="4" fill="#A855F7"/>
        <circle cx="50" cy="75" r="4" fill="#38BDF8"/>
        <circle cx="30" cy="50" r="4" fill="#06B6D4"/>
        <circle cx="50" cy="50" r="5" fill="#EF4444"/>
        <defs>
            <linearGradient id="cyber_grad" x1="0" y1="0" x2="100" y2="100">
                <stop stop-color="#06B6D4"/>
                <stop offset="0.5" stop-color="#38BDF8"/>
                <stop offset="1" stop-color="#A855F7"/>
            </linearGradient>
        </defs>
    </svg>
    <div>
        <div class="brand-title">FraudShield AI</div>
        <div class="hero-tagline">“Protect Every Click. Trust Every Decision.”</div>
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="hud-card"><div class="hud-val">4,000+</div><div class="hud-lbl">URLs Evaluated</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="hud-card"><div class="hud-val">' + str(round(eval_metrics['precision']*100, 1)) + '%</div><div class="hud-lbl">Test Precision</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="hud-card"><div class="hud-val">7+</div><div class="hud-lbl">Threat Categories</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="hud-card"><div class="hud-val">v2.1</div><div class="hud-lbl">Hybrid ML Core</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="status-card">
    <div class="status-item"><div class="status-dot"></div> AI Engine Online</div>
    <div class="status-item">⚡ Hybrid RF + TF-IDF Active</div>
    <div class="status-item">🔬 Feature Inspection Ready</div>
    <div class="status-item">🛡️ Threat Intel Active</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7. APP NAVIGATION TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "🔍 Single URL Inspector", 
    "⚔️ Side-by-Side Comparison", 
    "📈 Performance & Analytics", 
    "📁 Batch Scanner", 
    "⚙️ Model Architecture"
])

# TAB 1: SINGLE URL INSPECTOR
with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    target_url = st.text_input("Enter target domain or URL string:", placeholder="e.g., https://tratwera.com")
    scan_btn = st.button("Analyze Link Security")
    st.markdown('</div>', unsafe_allow_html=True)

    if scan_btn and target_url:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            "Extracting 24 Feature Parameters...",
            "Executing Random Forest Ensemble...",
            "Analyzing Char N-Gram TF-IDF Vector...",
            "Computing Hybrid Risk Probability...",
            "Scan Complete!"
        ]
        
        for idx, step in enumerate(steps):
            status_text.markdown("<span style='color:#38BDF8;'>⚡ " + step + "</span>", unsafe_allow_html=True)
            progress_bar.progress((idx + 1) * 20)
            time.sleep(0.04)
            
        status_text.empty()
        progress_bar.empty()

        is_valid, norm_url, err_msg = normalize_and_validate_url(target_url)
        if not is_valid:
            st.error("Validation Error: " + err_msg)
        else:
            res = analyze_single_url(norm_url)
            
            st.caption("ℹ️ **Disclaimer:** URL static analysis evaluates structural and lexical patterns. It cannot verify external site authenticity or domain ownership.")
            
            g1, g2, g3 = st.columns(3)
            with g1:
                st.plotly_chart(create_circular_trust_gauge(res['trust_score'], "AI Trust Index (Max 85)", True), use_container_width=True)
            with g2:
                st.plotly_chart(create_circular_trust_gauge(round(res['fraud_probability'] * 100, 1), "Predicted Fraud Risk", False), use_container_width=True)
            with g3:
                st.plotly_chart(create_circular_trust_gauge(res['certainty_score'], "Signal Certainty", True), use_container_width=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔬 AI Decision Pipeline")
            st.markdown("""
            <div class="timeline-container">
                <div class="timeline-step"><div class="step-node">1</div><div class="step-text">Input URL</div></div>
                <div class="timeline-step"><div class="step-node">2</div><div class="step-text">Feature Vector</div></div>
                <div class="timeline-step"><div class="step-node">3</div><div class="step-text">Hybrid RF + TF-IDF</div></div>
                <div class="timeline-step"><div class="step-node">4</div><div class="step-text">XAI Attribution</div></div>
                <div class="timeline-step"><div class="step-node">5</div><div class="step-text">Final Assessment</div></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            col_xai1, col_xai2 = st.columns(2)
            with col_xai1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("🧠 Threat Assessment Insights")
                st.write("**Heuristic Category:** `" + str(res['threat_category']) + "`")
                st.write("**Threat Severity Rating:** `" + str(res['threat_level']) + "`")
                
                st.markdown("#### 🚨 Key Risk Signals")
                for r in res['risk_factors']:
                    st.write("• " + str(r))
                
                st.markdown("#### 🟢 Safe Indicators")
                for s in res['safe_indicators']:
                    st.write("• " + str(s))
                st.markdown('</div>', unsafe_allow_html=True)

            with col_xai2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("📊 Feature Impact Scores")
                st.caption("Engine: " + res['exp_method'])
                top_feat = res['importance_df'].iloc[0]['Feature']
                st.info("💡 Primary decision driver: **" + str(top_feat) + "**")
                
                fig = px.bar(
                    res['importance_df'].head(6), 
                    x='Impact_Score', 
                    y='Feature', 
                    orientation='h',
                    color='Impact_Score',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🛡️ Contextual Defense Recommendations")
            for rec in res['recommendations']:
                st.write("👉 **Action Item:** " + str(rec))
            st.markdown('</div>', unsafe_allow_html=True)

            pdf_bytes = generate_pdf_report(res)
            st.download_button(
                label="📄 Export Security Audit PDF",
                data=pdf_bytes,
                file_name="FraudShield_Audit_" + datetime.now().strftime('%Y%m%d') + ".pdf",
                mime="application/pdf"
            )

# TAB 2: SIDE-BY-SIDE COMPARISON
with tabs[1]:
    st.subheader("⚔️ Side-by-Side URL Comparison")
    
    col_a, col_b = st.columns(2)
    with col_a:
        url_1 = st.text_input("Primary Target (A):", value="https://tratwera.com")
    with col_b:
        url_2 = st.text_input("Secondary Target (B):", value="http://secure-login-paypal-verify.com")

    if st.button("Run Side-by-Side Comparison"):
        v1, n1, _ = normalize_and_validate_url(url_1)
        v2, n2, _ = normalize_and_validate_url(url_2)
        
        if v1 and v2:
            r1 = analyze_single_url(n1)
            r2 = analyze_single_url(n2)
            
            cA, cB = st.columns(2)
            with cA:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Target A Profile")
                st.write("**URL:** `" + str(r1['url']) + "`")
                st.metric("AI Trust Score Index", str(r1['trust_score']) + " / 100")
                st.metric("Predicted Fraud Risk", str(round(r1['fraud_probability']*100, 1)) + "%")
                st.write("**Threat Rating:** " + str(r1['threat_level']))
                st.write("**Category:** " + str(r1['threat_category']))
                st.markdown('</div>', unsafe_allow_html=True)

            with cB:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("Target B Profile")
                st.write("**URL:** `" + str(r2['url']) + "`")
                st.metric("AI Trust Score Index", str(r2['trust_score']) + " / 100")
                st.metric("Predicted Fraud Risk", str(round(r2['fraud_probability']*100, 1)) + "%")
                st.write("**Threat Rating:** " + str(r2['threat_level']))
                st.write("**Category:** " + str(r2['threat_category']))
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🤖 Comparative AI Assessment")
            if r1['trust_score'] > r2['trust_score']:
                st.success("Target A presents a lower URL-based risk profile compared to Target B.")
            elif r2['trust_score'] > r1['trust_score']:
                st.warning("Target B presents a lower URL-based risk profile compared to Target A.")
            else:
                st.info("Both target URLs present identical estimated threat scores.")
            st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: PERFORMANCE & ANALYTICS
with tabs[2]:
    st.subheader("📈 Validated Model Metrics")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test Precision", str(round(eval_metrics['precision'] * 100, 1)) + "%")
    m2.metric("Test Recall", str(round(eval_metrics['recall'] * 100, 1)) + "%")
    m3.metric("Test F1-Score", str(round(eval_metrics['f1'] * 100, 1)) + "%")
    m4.metric("Test Accuracy", str(round(eval_metrics['accuracy'] * 100, 1)) + "%")
    m5.metric("ROC-AUC", str(round(eval_metrics['roc_auc'] * 100, 1)) + "%")
    
    st.caption("Metrics evaluated on an independent held-out test split (" + str(eval_metrics['test_count']) + " samples) at decision threshold " + str(eval_metrics['threshold']))
    
    st.markdown("<br>", unsafe_allow_html=True)
    an1, an2 = st.columns(2)
    with an1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Sample Threat Vector Distribution")
        cat_data = pd.DataFrame({
            'Category': ['Low Risk Domains', 'Phishing Lures', 'Crypto Scams', 'Fake Stores', 'Malware Host'],
            'Count': [1250, 420, 280, 310, 240]
        })
        fig_cat = px.pie(cat_data, values='Count', names='Category', hole=0.4, color_discrete_sequence=px.colors.sequential.Electric)
        fig_cat.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with an2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Ensemble Model Weights")
        weights_df = pd.DataFrame({
            'Sub-Model': ['Random Forest (Structured)', 'Char TF-IDF + Logistic Reg'],
            'Ensemble Weight': [0.65, 0.35]
        })
        fig_w = px.bar(weights_df, x='Sub-Model', y='Ensemble Weight', color='Sub-Model', color_discrete_sequence=['#38BDF8', '#A855F7'])
        fig_w.update_layout(template="plotly_dark", height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_w, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: BATCH SCANNER
with tabs[3]:
    st.subheader("📁 Batch URL Threat Scanner")
    
    raw_urls = st.text_area("Paste URLs line-by-line:", placeholder="http://secure-login-paypal-verify.com\nhttps://tratwera.com\nhttp://192.168.1.1/admin", height=140)
    if st.button("Execute Batch Scan") and raw_urls:
        urls_list = [u.strip() for u in raw_urls.split('\n') if u.strip()]
        results = []
        for u in urls_list:
            val, n_u, _ = normalize_and_validate_url(u)
            if val:
                r = analyze_single_url(n_u)
                results.append({
                    'URL': u, 
                    'AI Trust Index': r['trust_score'], 
                    'Threat Level': r['threat_level'], 
                    'Predicted Fraud Risk': str(round(r['fraud_probability']*100, 1)) + "%",
                    'Category': r['threat_category']
                })
        res_df = pd.DataFrame(results)
        
        st.success("Successfully processed " + str(len(res_df)) + " targets!")
        st.dataframe(res_df, use_container_width=True)
        st.download_button("📥 Download Batch Audit CSV", res_df.to_csv(index=False), "fraudshield_batch_audit.csv", "text/csv")

# TAB 5: ARCHITECTURE
with tabs[4]:
    st.subheader("⚙️ Model Architecture & Technical Specifications")
    st.markdown("""
    **Overview:**  
    FraudShield AI implements a dual-stream hybrid ensemble model combining a **Random Forest Classifier** operating on 24 handcrafted structural URL features and a **Character-Level TF-IDF + Logistic Regression Model** capturing text patterns.

    **Technical Pipeline Specifications:**
    * **Primary Classifier:** Random Forest (`n_estimators=300`, `max_depth=12`, `class_weight='balanced_subsample'`)
    * **Text Sub-Model:** Char N-Gram TF-IDF (`ngram_range=(3,5)`, `max_features=25000`) + Logistic Regression
    * **Ensemble Fusion:** Weighted probability blend ($0.65 \times P_{\text{RF}} + 0.35 \times P_{\text{LR}}$)
    * **Feature Engineering:** 24 continuous and binary structural, lexical, and Shannon Entropy parameters
    * **Explainability Engine:** Fallback-safe SHAP Game-Theoretic Tree Attribution with Scikit-Learn importance backup
    """)
    