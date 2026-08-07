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
# 1. PAGE CONFIGURATION & PREMIUM STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI v2.0 — Cyber Threat Intelligence Engine",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_cyber_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@600;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0B0E14;
        color: #F1F5F9;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1380px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .bg-particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -1;
        overflow: hidden;
        background: 
            radial-gradient(circle at 15% 20%, rgba(0, 242, 254, 0.07) 0%, transparent 45%),
            radial-gradient(circle at 85% 80%, rgba(168, 85, 247, 0.07) 0%, transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(11, 14, 20, 0.98) 0%, #0B0E14 100%);
        pointer-events: none;
    }

    .glass-card {
        background: rgba(15, 22, 35, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 242, 254, 0.15);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 242, 254, 0.35);
        box-shadow: 0 12px 40px rgba(0, 242, 254, 0.12);
    }

    .brand-container {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 12px;
    }

    .brand-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3.2rem;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        line-height: 1.1;
    }
    
    .hero-tagline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #00F2FE;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #94A3B8;
        max-width: 900px;
        line-height: 1.6;
        margin-bottom: 25px;
    }

    .stat-card {
        background: rgba(20, 28, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 242, 254, 0.3);
    }
    .stat-number {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: #00F2FE;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 10px 16px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        color: #94A3B8;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0px 22px;
        border: none !important;
        background-color: transparent;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.22) 0%, rgba(168, 85, 247, 0.22) 100%) !important;
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.45) !important;
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.25);
    }

    .stButton>button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        color: #0B0E14;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.35);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 28px rgba(0, 242, 254, 0.5);
    }

    .timeline-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    .timeline-step {
        background: rgba(20, 28, 45, 0.8);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        flex: 1;
        min-width: 140px;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .timeline-arrow {
        color: #A855F7;
        font-weight: bold;
        font-size: 1.2rem;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(10, 185, 129, 0.3);
    }
    </style>
    <div class="bg-particles"></div>
    """, unsafe_allow_html=True)

inject_cyber_theme()

# -----------------------------------------------------------------------------
# 2. URL NORMALIZATION & FEATURE EXTRACTION
# -----------------------------------------------------------------------------
def normalize_and_validate_url(url_str: str) -> tuple[bool, str, str]:
    if not url_str or not isinstance(url_str, str):
        return False, "", "Empty URL provided."
    clean_url = url_str.strip().lower()
    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        clean_url = "http://" + clean_url
    try:
        parsed = urlparse(clean_url)
        domain = parsed.netloc if parsed.netloc else parsed.path.split('/')[0]
        ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        if not domain or (not ip_pattern.match(domain.split(':')[0]) and '.' not in domain):
            return False, clean_url, "Invalid URL structure."
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
# 3. ML MODEL & SHAP CACHING ENGINE
# -----------------------------------------------------------------------------
MODEL_FILE = "fraudshield_rf_model.joblib"

def generate_synthetic_dataset(n_samples: int = 2500) -> pd.DataFrame:
    np.random.seed(42)
    data = []
    for _ in range(n_samples // 2):
        data.append([int(np.random.normal(28, 8)), int(np.random.normal(12, 4)), 1, 0, 0, 2, int(np.random.poisson(1)), 20, 1, 0, 0, 0, 0, float(np.random.normal(3.8, 0.4)), 0, 0, 0])
    for _ in range(n_samples // 2):
        data.append([int(np.random.normal(75, 20)), int(np.random.normal(25, 8)), 3, 2, 1, 4, int(np.random.normal(12, 5)), 35, 0, 0, 2, 2, 4, float(np.random.normal(4.8, 0.5)), 0, 0, 1])
    cols = FEATURE_NAMES + ['is_fraud']
    return pd.DataFrame(data, columns=cols).clip(lower=0)

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass
    df = generate_synthetic_dataset()
    X, y = df[FEATURE_NAMES], df['is_fraud']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    return model

model = load_or_train_model()

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

def classify_threat_category(url: str, features: dict, fraud_prob: float) -> str:
    if fraud_prob < 0.35:
        return "Legitimate / Safe"
    url_lower = url.lower()
    if any(k in url_lower for k in ['paypal', 'bank', 'secure', 'login', 'auth', 'verify', 'account']):
        return "Credential Theft / Phishing"
    elif any(k in url_lower for k in ['shop', 'store', 'cart', 'discount', 'checkout']):
        return "Fake Shopping Website"
    elif any(k in url_lower for k in ['crypto', 'wallet', 'binance', 'btc', 'claim']):
        return "Crypto Scam"
    elif features['is_ip'] == 1 or features['entropy'] > 4.9:
        return "Malware / Exploit Website"
    else:
        return "Phishing Website"

def generate_contextual_recommendations(category: str, features: dict) -> tuple[list, list]:
    reasons = []
    recs = []

    if features['has_https'] == 0:
        reasons.append("Unencrypted connection (HTTP protocol detected).")
        recs.append("Never enter passwords, credit cards, or personal details on HTTP sites.")
    if features['is_ip'] == 1:
        reasons.append("Domain points directly to a raw IP address.")
        recs.append("Avoid accessing raw IP addresses unless verifying internal network devices.")
    if features['url_length'] > 65:
        reasons.append("Excessively long URL structure (" + str(features['url_length']) + " characters).")
        recs.append("Inspect the core domain carefully in the browser address bar.")
    if features['suspicious_keywords_count'] > 0:
        reasons.append("Contains " + str(features['suspicious_keywords_count']) + " high-risk lure keywords.")

    if category == "Credential Theft / Phishing":
        recs.append("Verify the domain on official bookmarks before logging in.")
        recs.append("Enable Multi-Factor Authentication (MFA) on your account immediately.")
    elif category == "Crypto Scam":
        recs.append("Never connect Web3 wallets or sign approval transactions on unverified links.")
        recs.append("Verify crypto giveaway claims on official verified social media channels.")
    elif category == "Fake Shopping Website":
        recs.append("Check domain registration age and look for missing merchant contact info.")
        recs.append("Use secure credit card payment methods with fraud protection.")
    elif category == "Malware / Exploit Website":
        recs.append("Do not download any files or accept pop-up prompt installations.")
        recs.append("Run a full antivirus scan if you accessed this website.")
    else:
        if not recs:
            recs.append("Standard domain metrics detected with no structural anomalies.")
            recs.append("Practice normal online caution.")

    if not reasons:
        reasons.append("Structural URL parameters fall within safe statistical boundaries.")

    return reasons, recs

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
    
    reasons, recs = generate_contextual_recommendations(threat_category, feats)

    return {
        'url': url, 'fraud_probability': fraud_prob, 'trust_score': trust_score,
        'confidence_score': confidence_score, 'threat_level': threat_level, 
        'threat_category': threat_category, 'features': feats, 
        'shap_importance': shap_df, 'reasons': reasons, 'recommendations': recs
    }

# -----------------------------------------------------------------------------
# 4. PLOTLY GAUGES & VISUALIZATIONS
# -----------------------------------------------------------------------------
def create_score_gauge(score: float, title: str, is_trust: bool = True) -> go.Figure:
    color = "#10B981" if (score >= 70 if is_trust else score < 30) else ("#F59E0B" if (score >= 40 if is_trust else score < 60) else "#EF4444")
    suffix = "%" if not is_trust else "/100"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': suffix, 'font': {'color': color, 'size': 28, 'family': 'Orbitron'}},
        title={'text': title, 'font': {'size': 14, 'color': '#94A3B8', 'family': 'Space Grotesk'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
            'bar': {'color': color},
            'bgcolor': "rgba(15, 23, 42, 0.5)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
        }
    ))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# -----------------------------------------------------------------------------
# 5. ENTERPRISE PDF REPORT GENERATOR (FIXED STRING FORMATTING)
# -----------------------------------------------------------------------------
def generate_pdf_report(analysis: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#0B0E14'), spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#64748B'), spaceAfter=15)
    section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13, textColor=colors.HexColor('#0284C7'), spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=14, textColor=colors.HexColor('#334155'))

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sub_text = "Generated: " + now_str + " UTC | Target: " + str(analysis['url'])

    story = [
        Paragraph("FraudShield AI — Cyber Threat Audit Report", title_style),
        Paragraph(sub_text, subtitle_style),
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284C7'), spaceAfter=15),
        Paragraph("Executive Threat Intelligence Summary", section_heading)
    ]

    summary_data = [
        ["Target Domain / URL", str(analysis['url'])],
        ["Overall Threat Level", str(analysis['threat_level'])],
        ["Trust Score Index", str(analysis['trust_score']) + " / 100"],
        ["Fraud Probability", str(round(analysis['fraud_probability'] * 100, 1)) + "%"],
        ["Model Confidence Score", str(analysis['confidence_score']) + "%"],
        ["Assigned Threat Category", str(analysis['threat_category'])]
    ]
    
    t = Table(summary_data, colWidths=[150, 370])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Risk Indicators & Contextual Action Items", section_heading))
    for reason in analysis['reasons']:
        story.append(Paragraph("• <b>Identified Risk Factor:</b> " + str(reason), body_style))
    story.append(Spacer(1, 6))
   