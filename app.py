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
# 1. PAGE CONFIGURATION & ANIMATED BACKGROUND CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI — Trust Before You Click",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_cyber_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Orbitron:wght@500;700;900&family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #07090E;
        color: #E2E8F0;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1280px;
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
        background: radial-gradient(circle at 20% 20%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 80% 80%, rgba(168, 85, 247, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 50% 50%, rgba(11, 14, 20, 0.95) 0%, #07090E 100%);
        pointer-events: none;
    }

    .glass-card {
        background: rgba(15, 21, 33, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }

    .brand-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 3rem;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        line-height: 1.1;
    }
    
    .brand-tagline {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        color: #94A3B8;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 8px 14px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        border-radius: 10px;
        color: #94A3B8;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0px 18px;
        border: none !important;
        background-color: transparent;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%) !important;
        color: #00F2FE !important;
        border: 1px solid rgba(0, 242, 254, 0.4) !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.25);
    }

    .stButton>button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        color: #07090E;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
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
        return False, clean_url, f"Malformed URL: {str(e)}"

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

def generate_synthetic_dataset(n_samples: int = 2000) -> pd.DataFrame:
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
    
    threat_level = "Safe" if fraud_prob < 0.20 else ("Low Risk" if fraud_prob < 0.40 else ("Medium Risk" if fraud_prob < 0.65 else ("High Risk" if fraud_prob < 0.85 else "Critical Risk")))
    threat_category = classify_threat_category(url, feats, fraud_prob)
    shap_df = compute_shap_explanations(feat_df)
    
    reasons, recs = [], []
    if feats['has_https'] == 0:
        reasons.append("Missing HTTPS protocol — communication with this site is unencrypted.")
        recs.append("Never enter passwords or financial information on HTTP sites.")
    if feats['is_ip'] == 1:
        reasons.append("Raw IP address used instead of a domain name.")
        recs.append("Avoid accessing websites hosted directly on IP addresses.")
    if feats['url_length'] > 60:
        reasons.append(f"Excessively long URL ({feats['url_length']} chars) attempting to mask destination.")
        recs.append("Inspect the root domain carefully in your address bar.")
    if feats['suspicious_keywords_count'] > 0:
        reasons.append(f"Detected {feats['suspicious_keywords_count']} high-risk scam keywords.")
        recs.append("Do not click unexpected links offering rewards or urgent resets.")

    if not reasons:
        reasons.append("Standard domain metrics detected with no structural anomalies.")
        recs.append("Website exhibits standard behavioral markers. Normal online caution applies.")

    return {
        'url': url, 'fraud_probability': fraud_prob, 'trust_score': trust_score,
        'threat_level': threat_level, 'threat_category': threat_category,
        'features': feats, 'shap_importance': shap_df, 'reasons': reasons, 'recommendations': recs
    }

def generate_pdf_report(analysis: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=20, textColor=colors.HexColor('#0F172A'), spaceAfter=4)
    subtitle_style = ParagraphStyle('DocSubtitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#64748B'), spaceAfter=15)
    section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=colors.HexColor('#0284C7'), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=9, leading=13, textColor=colors.HexColor('#334155'))

    story = [
        Paragraph("FraudShield AI — Security Audit Report", title_style),
        Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Target: {analysis['url']}", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0'), spaceAfter=15),
        Paragraph("Executive Threat Summary", section_heading)
    ]

    summary_data = [
        ["Target URL", analysis['url']],
        ["Threat Level", analysis['threat_level']],
        ["Trust Score", f"{analysis['trust_score']} / 100"],
        ["Fraud Probability", f"{round(analysis['fraud_probability'] * 100, 1)}%"],
        ["Threat Category", analysis['threat_category']]
    ]
    
    t = Table(summary_data, colWidths=[140, 380])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1E293B')),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Risk Indicators & Recommendations", section_heading))
    for reason in analysis['reasons']:
        story.append(Paragraph(f"• <b>Risk Factor:</b> {reason}", body_style))
    for rec in analysis['recommendations']:
        story.append(Paragraph(f"• <b>Action:</b> {rec}", body_style))

    doc.build(story)
    return buffer.getvalue()

# -----------------------------------------------------------------------------
# 4. USER INTERFACE LAYOUT
# -----------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=200&auto=format&fit=crop&q=60", width=85)
with col_title:
    st.markdown('<div class="brand-title">FraudShield AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-tagline">Trust Before You Click — Cyber Threat Intelligence Engine</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tabs = st.tabs(["🔍 URL Inspector", "📁 Batch Analysis", "⚙️ Model Intelligence"])

with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    target_url = st.text_input("Enter URL to analyze:", placeholder="e.g., http://secure-login-paypal-verify.com/login")
    scan_btn = st.button("Analyze Link Security")
    st.markdown('</div>', unsafe_allow_html=True)

    if scan_btn and target_url:
        with st.spinner("Executing lexical extraction & ML classification..."):
            is_valid, norm_url, err_msg = normalize_and_validate_url(target_url)
            if not is_valid:
                st.error(f"Validation Error: {err_msg}")
            else:
                res = analyze_single_url(norm_url)
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Trust Score", f"{res['trust_score']} / 100")
                with m2:
                    st.metric("Threat Level", res['threat_level'])
                with m3:
                    st.metric("Fraud Probability", f"{round(res['fraud_probability'] * 100, 1)}%")
                with m4:
                    st.metric("Category", res['threat_category'])

                st.markdown("<br>", unsafe_allow_html=True)
                
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    st.subheader("Key Risk Indicators")
                    for r in res['reasons']:
                        st.write(f"⚠️ {r}")
                    st.subheader("Recommended Actions")
                    for rec in res['recommendations']:
                        st.write(f"🛡️ {rec}")

                with c_right:
                    st.subheader("SHAP Feature Impact Breakdown")
                    fig = px.bar(
                        res['shap_importance'].head(7), 
                        x='SHAP_Impact', 
                        y='Feature', 
                        orientation='h',
                        color='SHAP_Impact',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(template="plotly_dark", height=280)
                    st.plotly_chart(fig, use_container_width=True)

                pdf_bytes = generate_pdf_report(res)
                st.download_button(
                    label="📄 Download Full PDF Security Audit",
                    data=pdf_bytes,
                    file_name=f"FraudShield_Audit_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

with tabs[1]:
    st.subheader("Batch URL Threat Scanner")
    uploaded_file = st.file_uploader("Upload CSV containing 'url' column:", type=['csv'])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if 'url' in batch_df.columns:
            if st.button("Process Batch URLs"):
                results = []
                for u in batch_df['url']:
                    val, n_u, _ = normalize_and_validate_url(str(u))
                    if val:
                        r = analyze_single_url(n_u)
                        results.append({'URL': u, 'Trust Score': r['trust_score'], 'Threat Level': r['threat_level'], 'Fraud Prob': r['fraud_probability']})
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)
                st.download_button("📥 Export Results as CSV", res_df.to_csv(index=False), "fraudshield_batch_results.csv", "text/csv")
        else:
            st.error("Uploaded CSV must contain a 'url' column.")

with tabs[2]:
    st.subheader("Random Forest Model Architecture")
    st.json({
        "Model Type": "RandomForestClassifier",
        "Estimators": 100,
        "Max Depth": 10,
        "Features Extracted": len(FEATURE_NAMES),
        "Class Balance Weighting": "Balanced"
    })
    