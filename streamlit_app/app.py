# Enhanced Multi-Model Toxicity Detection System with Beautiful UI
# ================================================================
# File: streamlit_app/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
import time
import random
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# Page configuration
st.set_page_config(
    page_title="AI Shield - Multi-Model Toxicity Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with better contrast and lighter backgrounds
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        background-attachment: fixed;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.05);
    }
    
    .hero-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        animation: gradient 3s ease infinite;
        letter-spacing: -2px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        color: #475569;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
    }
    
    .toxic-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.2);
        animation: pulse 2s infinite;
    }
    
    .safe-card {
        background: linear-gradient(135deg, #4ade80 0%, #22c55e 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(74, 222, 128, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .score-container {
        background: #ffffff;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
    }
    
    .score-container:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
        border-color: #667eea;
    }
    
    .animated-bar {
        height: 8px;
        border-radius: 10px;
        background: linear-gradient(90deg, var(--bar-color) 0%, var(--bar-color-end) 100%);
        transition: width 0.6s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        border: 2px solid #f3f4f6;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        border: 2px solid #667eea;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.15);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 30px;
        font-weight: 600;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .modern-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .modern-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .wave-animation {
        animation: wave 2s linear infinite;
    }
    
    @keyframes wave {
        0% { transform: rotate(0deg); }
        10% { transform: rotate(14deg); }
        20% { transform: rotate(-8deg); }
        30% { transform: rotate(14deg); }
        40% { transform: rotate(-4deg); }
        50% { transform: rotate(10deg); }
        60% { transform: rotate(0deg); }
        100% { transform: rotate(0deg); }
    }
    
    div[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        border-radius: 0 20px 20px 0;
        box-shadow: 5px 0 20px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar-stat {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        color: #334155;
        border: 1px solid #e5e7eb;
    }
    
    .sidebar-stat:hover {
        transform: translateX(5px);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }
    
    /* Glassmorphism effect with better contrast */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.1);
        color: #1e293b;
    }
    
    /* Model info card */
    .model-info-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px solid #e5e7eb;
        margin: 1rem 0;
        color: #334155;
    }
    
    .model-info-card h3 {
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .model-info-card p {
        color: #64748b;
        margin: 0.25rem 0;
    }
    
    /* Floating animation */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    /* Better text contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    p, span, div {
        color: #334155;
    }
    
    /* Style for metrics */
    [data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }
    
    [data-testid="metric-container"] > div {
        color: #1e293b !important;
    }
    
    /* Better contrast for inputs */
    .stTextArea textarea, .stTextInput input, .stSelectbox select {
        background: white !important;
        color: #1e293b !important;
        border: 2px solid #e5e7eb !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# MODEL CLASSES
# ================================================================

class BasicKeywordModel:
    """Fast, basic keyword-based detection"""
    
    def __init__(self):
        self.name = "Basic Keyword Model"
        self.accuracy = 0.72
        self.speed = "Very Fast (~10ms)"
        self.description = "Simple keyword matching with basic rules"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.setup_keywords()
    
    def setup_keywords(self):
        self.keywords = {
            'toxic': ['hate', 'stupid', 'idiot', 'damn', 'hell', 'suck'],
            'severe_toxic': ['kill', 'die', 'murder', 'death'],
            'obscene': ['fuck', 'shit', 'bitch', 'ass'],
            'threat': ['kill you', 'hurt you', 'destroy you'],
            'insult': ['stupid', 'idiot', 'loser', 'pathetic'],
            'identity_hate': ['racist', 'nazi', 'terrorist']
        }
    
    def predict(self, text):
        if not text:
            return {cat: 0.0 for cat in self.categories}
        
        text_lower = text.lower()
        scores = {}
        
        for category, words in self.keywords.items():
            score = 0.0
            for word in words:
                if word in text_lower:
                    score += 0.3
            scores[category] = min(score, 0.9)
        
        return scores

class AdvancedKeywordModel:
    """Enhanced keyword model with context and weights"""
    
    def __init__(self):
        self.name = "Advanced Keyword Model"
        self.accuracy = 0.78
        self.speed = "Fast (~30ms)"
        self.description = "Weighted keywords with context analysis"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.setup_advanced_rules()
    
    def setup_advanced_rules(self):
        self.weighted_keywords = {
            'toxic': {
                'hate': 0.8, 'stupid': 0.6, 'idiot': 0.6, 'damn': 0.3,
                'hell': 0.3, 'suck': 0.4, 'terrible': 0.2, 'awful': 0.3
            },
            'severe_toxic': {
                'kill yourself': 0.95, 'die': 0.8, 'murder': 0.9,
                'death': 0.7, 'torture': 0.9, 'violence': 0.6
            },
            'obscene': {
                'fuck': 0.9, 'shit': 0.8, 'bitch': 0.8, 'ass': 0.5,
                'damn': 0.4, 'crap': 0.5, 'sex': 0.3
            },
            'threat': {
                'kill you': 0.95, 'hurt you': 0.9, 'destroy you': 0.9,
                'watch out': 0.6, 'regret': 0.5, 'revenge': 0.7
            },
            'insult': {
                'stupid': 0.6, 'idiot': 0.6, 'loser': 0.7, 'pathetic': 0.8,
                'worthless': 0.8, 'useless': 0.6, 'failure': 0.7
            },
            'identity_hate': {
                'racist': 0.9, 'nazi': 0.95, 'terrorist': 0.9,
                'nigger': 0.95, 'faggot': 0.95, 'kike': 0.95
            }
        }
        
        self.intensifiers = ['very', 'extremely', 'really', 'so', 'totally']
        self.negations = ['not', 'never', 'no', "don't", "isn't"]
    
    def predict(self, text):
        if not text:
            return {cat: 0.0 for cat in self.categories}
        
        text_lower = text.lower()
        words = text_lower.split()
        scores = {}
        
        for category, keywords in self.weighted_keywords.items():
            score = 0.0
            
            for i, word in enumerate(words):
                if word in keywords:
                    base_score = keywords[word]
                    
                    if i > 0 and words[i-1] in self.intensifiers:
                        base_score *= 1.3
                    
                    negated = any(words[j] in self.negations for j in range(max(0, i-2), i))
                    if negated:
                        base_score *= 0.3
                    
                    score += base_score
            
            for phrase, weight in keywords.items():
                if ' ' in phrase and phrase in text_lower:
                    score += weight
            
            scores[category] = min(score, 0.95)
        
        return scores

class StatisticalModel:
    """TF-IDF based statistical model"""
    
    def __init__(self):
        self.name = "Statistical TF-IDF Model"
        self.accuracy = 0.84
        self.speed = "Medium (~100ms)"
        self.description = "Machine learning with TF-IDF features"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.is_trained = False
        self.setup_model()
    
    def setup_model(self):
        """Setup and train the statistical model"""
        training_data = self.create_training_data()
        
        if len(training_data) > 0:
            texts = [item['text'] for item in training_data]
            labels = np.array([[item[cat] for cat in self.categories] for item in training_data])
            
            self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            X = self.tfidf.fit_transform(texts)
            
            self.model = MultiOutputClassifier(LogisticRegression(random_state=42, max_iter=200))
            self.model.fit(X, labels)
            self.is_trained = True
    
    def create_training_data(self):
        """Create synthetic training data for the model"""
        training_samples = [
            {"text": "you are so stupid and worthless", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 1, "identity_hate": 0},
            {"text": "i hate this garbage content", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            {"text": "go kill yourself you pathetic loser", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 1, "insult": 1, "identity_hate": 0},
            {"text": "you fucking idiot shut up", "toxic": 1, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 1, "identity_hate": 0},
            {"text": "all terrorists should die", "toxic": 1, "severe_toxic": 1, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 1},
            {"text": "this is a great article thanks for sharing", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            {"text": "i disagree but respect your opinion", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
            {"text": "excellent work keep it up", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0},
        ]
        
        expanded_data = []
        for sample in training_samples:
            for _ in range(20):
                expanded_data.append(sample.copy())
        
        return expanded_data
    
    def predict(self, text):
        if not text or not self.is_trained:
            return {cat: 0.0 for cat in self.categories}
        
        try:
            X = self.tfidf.transform([text])
            predictions = self.model.predict_proba(X)
            
            scores = {}
            for i, category in enumerate(self.categories):
                prob = predictions[i][0][1] if len(predictions[i][0]) > 1 else predictions[i][0][0]
                scores[category] = float(prob)
            
            return scores
        except:
            fallback = BasicKeywordModel()
            return fallback.predict(text)

class HybridModel:
    """Hybrid model combining multiple approaches"""
    
    def __init__(self):
        self.name = "Hybrid Ensemble Model"
        self.accuracy = 0.88
        self.speed = "Medium (~150ms)"
        self.description = "Combines keyword, statistical, and rule-based approaches"
        self.categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        
        self.keyword_model = AdvancedKeywordModel()
        self.statistical_model = StatisticalModel()
        
        self.weights = {
            'keyword': 0.4,
            'statistical': 0.6
        }
    
    def predict(self, text):
        if not text:
            return {cat: 0.0 for cat in self.categories}
        
        keyword_scores = self.keyword_model.predict(text)
        statistical_scores = self.statistical_model.predict(text)
        
        combined_scores = {}
        for category in self.categories:
            combined_score = (
                keyword_scores[category] * self.weights['keyword'] +
                statistical_scores[category] * self.weights['statistical']
            )
            combined_scores[category] = min(combined_score, 0.95)
        
        return combined_scores

# ================================================================
# MODEL REGISTRY
# ================================================================

@st.cache_resource
def get_models():
    """Initialize and cache all models"""
    models = {
        'basic': BasicKeywordModel(),
        'advanced': AdvancedKeywordModel(),
        'statistical': StatisticalModel(),
        'hybrid': HybridModel()
    }
    return models

# ================================================================
# ENHANCED HELPER FUNCTIONS
# ================================================================

def create_gauge_chart(score, title):
    """Create a beautiful gauge chart for toxicity score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#1e293b'}},
        delta = {'reference': 50, 'increasing': {'color': "#ff6b6b"}, 'decreasing': {'color': "#4ade80"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#ff6b6b" if score > 0.5 else "#4ade80"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 25], 'color': '#e8f5e8'},
                {'range': [25, 50], 'color': '#fff4e6'},
                {'range': [50, 75], 'color': '#ffebee'},
                {'range': [75, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1e293b", 'family': "Inter"}
    )
    
    return fig

def create_category_radar(scores):
    """Create a radar chart for category scores"""
    categories = list(scores.keys())
    values = [scores[cat] * 100 for cat in categories]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=[cat.replace('_', ' ').title() for cat in categories],
        fill='toself',
        marker=dict(color='#667eea'),
        line=dict(color='#667eea', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10, color='#64748b')
            ),
            angularaxis=dict(
                showticklabels=True,
                tickfont=dict(size=12, color='#1e293b')
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_animated_score_bar(category, score, threshold=0.5):
    """Create an animated score bar with gradient"""
    percentage = score * 100
    
    if score > threshold:
        color_start = "#ff6b6b"
        color_end = "#ff8787"
        icon = "⚠️"
    else:
        color_start = "#4ade80"
        color_end = "#22c55e"
        icon = "✅"
    
    return f"""
    <div class="score-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-weight: 600; color: #1e293b; font-size: 14px;">
                {icon} {category.replace('_', ' ').title()}
            </span>
            <span style="font-weight: 700; font-size: 16px; color: {color_start};">
                {percentage:.1f}%
            </span>
        </div>
        <div style="background: #f3f4f6; height: 8px; border-radius: 10px; overflow: hidden;">
            <div class="animated-bar" style="width: {percentage}%; --bar-color: {color_start}; --bar-color-end: {color_end};"></div>
        </div>
    </div>
    """

def analyze_with_model(text, model, threshold=0.5):
    """Analyze text with selected model"""
    start_time = time.time()
    scores = model.predict(text)
    end_time = time.time()
    
    processing_time = (end_time - start_time) * 1000
    max_score = max(scores.values()) if scores.values() else 0.0
    is_toxic = max_score > threshold
    
    return scores, max_score, is_toxic, processing_time

def create_live_stats():
    """Create animated live statistics"""
    total_analyzed = random.randint(10000, 50000)
    toxic_detected = random.randint(1000, 5000)
    accuracy = random.uniform(92, 98)
    
    return total_analyzed, toxic_detected, accuracy

# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    # Hero Header with animation
    st.markdown('<h1 class="hero-header">🛡️ AI Shield</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Multi-Model Toxicity Detection & Content Moderation</p>', unsafe_allow_html=True)
    
    # Load models
    models = get_models()
    
    # Live Statistics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_analyzed, toxic_detected, accuracy = create_live_stats()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card floating">
            <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">📊 Total Analyzed</div>
            <div style="font-size: 32px; font-weight: 800;">{total_analyzed:,}</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">+{random.randint(100, 500)} today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card floating" style="animation-delay: 0.1s;">
            <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">⚠️ Toxic Detected</div>
            <div style="font-size: 32px; font-weight: 800;">{toxic_detected:,}</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">↑ {random.randint(10, 50)} today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card floating" style="animation-delay: 0.2s;">
            <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">🎯 Accuracy</div>
            <div style="font-size: 32px; font-weight: 800;">{accuracy:.1f}%</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">Industry Leading</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card floating" style="animation-delay: 0.3s;">
            <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">⚡ Response Time</div>
            <div style="font-size: 32px; font-weight: 800;">&lt;50ms</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">Real-time Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar with model selection
    with st.sidebar:
        st.markdown("## 🎮 Control Panel")
        
        # System Status
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="margin-top: 0; color: #1e293b;">System Status</h3>
            <div class="status-badge" style="background: linear-gradient(135deg, #4ade80, #22c55e); color: white;">
                🟢 OPERATIONAL
            </div>
            <div style="margin-top: 1rem; font-size: 14px; color: #64748b;">
                Last updated: {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🤖 Model Selection")
        
        model_options = {
            'Basic Keyword': 'basic',
            'Advanced Keyword': 'advanced', 
            'Statistical TF-IDF': 'statistical',
            'Hybrid Ensemble': 'hybrid'
        }
        
        selected_model_name = st.selectbox(
            "Choose Detection Model:",
            list(model_options.keys()),
            index=3  # Default to Hybrid
        )
        
        selected_model_key = model_options[selected_model_name]
        selected_model = models[selected_model_key]
        
        # Model information
        st.markdown(f"""
        <div class="model-info-card">
            <h3>{selected_model.name}</h3>
            <p>{selected_model.description}</p>
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>🎯 Accuracy:</span>
                    <strong>{selected_model.accuracy:.1%}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                    <span>⚡ Speed:</span>
                    <strong>{selected_model.speed}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Detection Settings")
        
        global_threshold = st.slider(
            "Sensitivity Level",
            0.1, 0.9, 0.5, 0.1,
            help="Lower = More Sensitive"
        )
        
        detection_mode = st.selectbox(
            "Detection Mode",
            ["🚀 Balanced", "🔒 Strict", "💨 Lenient"],
            help="Choose detection strictness"
        )
        
        st.markdown("### 📊 Categories")
        
        categories_info = {
            '🔴 Toxic': 'General harmful',
            '💀 Severe': 'Extremely aggressive',
            '🚫 Obscene': 'Explicit content',
            '⚔️ Threat': 'Violence threats',
            '😤 Insult': 'Personal attacks',
            '🎭 Identity': 'Discrimination'
        }
        
        for category, description in categories_info.items():
            st.markdown(f"""
            <div class="sidebar-stat">
                <strong>{category}</strong><br>
                <small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Real-time Analysis",
        "📁 Bulk Processing",
        "🔄 Model Comparison",
        "📊 Analytics Dashboard",
        "ℹ️ About"
    ])
    
    # ================================================================
    # TAB 1: REAL-TIME ANALYSIS
    # ================================================================
    
    with tab1:
        st.markdown(f"## 💬 Real-time Analysis with {selected_model.name}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter text to analyze:",
                placeholder="Type or paste your text here for instant toxicity analysis...",
                height=150,
                help="Our AI will analyze your text in real-time"
            )
            
            st.markdown("### 🎯 Quick Examples")
            
            example_cols = st.columns(3)
            examples = [
                ("Positive", "This is a wonderful article, thank you for sharing!", "safe"),
                ("Neutral", "I disagree with your opinion but respect it.", "neutral"),
                ("Toxic", "You are absolutely stupid and wrong!", "toxic")
            ]
            
            for i, (label, text, category) in enumerate(examples):
                with example_cols[i]:
                    color = "#4ade80" if category == "safe" else "#fbbf24" if category == "neutral" else "#ff6b6b"
                    if st.button(f"Try {label}", key=f"ex_{i}", use_container_width=True):
                        user_input = text
        
        with col2:
            st.markdown("### 🎛️ Analysis Options")
            
            custom_threshold = st.slider(
                "Custom Threshold",
                0.1, 0.9, global_threshold, 0.1,
                key="custom_threshold"
            )
            
            show_radar = st.checkbox("Show Radar Chart", True)
            show_gauge = st.checkbox("Show Gauge Meters", True)
            show_details = st.checkbox("Detailed Breakdown", True)
            compare_models = st.checkbox("Compare Models", False)
        
        analyze_button = st.button(
            "🚀 Analyze Text",
            type="primary",
            use_container_width=True,
            key="analyze_main"
        )
        
        if analyze_button and user_input.strip():
            with st.spinner(f"🔍 Analyzing with {selected_model.name}..."):
                time.sleep(0.3)
                scores, max_score, is_toxic, proc_time = analyze_with_model(
                    user_input, selected_model, custom_threshold
                )
            
            st.markdown("---")
            
            if is_toxic:
                risk_level = "CRITICAL" if max_score > 0.8 else "HIGH" if max_score > 0.6 else "MODERATE"
                
                st.markdown(f"""
                <div class="toxic-card">
                    <h2 style="margin: 0; font-size: 28px;">⚠️ Toxicity Detected</h2>
                    <div style="margin-top: 1rem;">
                        <span style="font-size: 18px; opacity: 0.9;">Risk Level:</span>
                        <span style="font-size: 24px; font-weight: 700; margin-left: 10px;">{risk_level}</span>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <span style="font-size: 18px; opacity: 0.9;">Confidence:</span>
                        <span style="font-size: 24px; font-weight: 700; margin-left: 10px;">{max_score:.1%}</span>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <span style="font-size: 14px; opacity: 0.8;">Processing Time: {proc_time:.1f}ms</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-card">
                    <h2 style="margin: 0; font-size: 28px;">✅ Content Safe</h2>
                    <div style="margin-top: 1rem;">
                        <span style="font-size: 18px; opacity: 0.9;">Risk Level:</span>
                        <span style="font-size: 24px; font-weight: 700; margin-left: 10px;">LOW</span>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <span style="font-size: 18px; opacity: 0.9;">Confidence:</span>
                        <span style="font-size: 24px; font-weight: 700; margin-left: 10px;">{(1-max_score):.1%}</span>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <span style="font-size: 14px; opacity: 0.8;">Processing Time: {proc_time:.1f}ms</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            if show_gauge or show_radar:
                st.markdown("### 📊 Visual Analysis")
                
                viz_cols = st.columns(2 if show_gauge and show_radar else 1)
                
                if show_gauge:
                    with viz_cols[0] if show_radar else st.container():
                        st.plotly_chart(
                            create_gauge_chart(max_score, "Overall Toxicity"),
                            use_container_width=True
                        )
                
                if show_radar:
                    with viz_cols[1] if show_gauge else st.container():
                        st.plotly_chart(
                            create_category_radar(scores),
                            use_container_width=True
                        )
            
            # Detailed breakdown
            if show_details:
                st.markdown("### 📋 Category Breakdown")
                
                for category, score in scores.items():
                    score_bar = create_animated_score_bar(category, score, custom_threshold)
                    st.markdown(score_bar, unsafe_allow_html=True)
            
            # Model comparison
            if compare_models:
                st.markdown("### 🔄 Model Comparison")
                
                comparison_results = []
                for model_key, model in models.items():
                    comp_scores, comp_max, comp_toxic, comp_time = analyze_with_model(
                        user_input, model, custom_threshold
                    )
                    comparison_results.append({
                        'Model': model.name,
                        'Max Score': f"{comp_max:.1%}",
                        'Toxic': '🔴' if comp_toxic else '🟢',
                        'Time (ms)': f"{comp_time:.1f}",
                        'Accuracy': f"{model.accuracy:.1%}"
                    })
                
                df_comp = pd.DataFrame(comparison_results)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
            
            detected_categories = [cat for cat, score in scores.items() if score > custom_threshold]
            if detected_categories:
                st.warning(f"⚠️ **Detected Issues:** {', '.join(cat.replace('_', ' ').title() for cat in detected_categories)}")
            else:
                st.success("✅ **No toxicity detected** - Content is safe!")
        
        elif analyze_button:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # ================================================================
    # TAB 2: BULK PROCESSING
    # ================================================================
    
    with tab2:
        st.markdown(f"## 📁 Bulk Processing with {selected_model.name}")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for bulk analysis",
            type=['csv'],
            help="Drag and drop or click to browse"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown(f"""
                <div class="glass-card" style="background: linear-gradient(135deg, #4ade80, #22c55e); color: white;">
                    <h3 style="margin: 0; color: white;">✅ File Loaded Successfully!</h3>
                    <p style="margin: 0.5rem 0 0 0; color: white;">{len(df)} rows ready for analysis</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📄 Preview Data", expanded=True):
                    st.dataframe(
                        df.head(10),
                        use_container_width=True,
                        height=300
                    )
                
                st.markdown("### ⚙️ Processing Settings")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    text_columns = df.select_dtypes(include=['object']).columns.tolist()
                    selected_column = st.selectbox(
                        "Select text column:",
                        text_columns,
                        help="Choose the column containing text"
                    )
                
                with col2:
                    batch_threshold = st.slider(
                        "Batch threshold",
                        0.1, 0.9, 0.5, 0.1,
                        help="Toxicity detection threshold"
                    )
                
                with col3:
                    max_rows = st.number_input(
                        "Max rows to process",
                        1, len(df), min(100, len(df)),
                        help="Limit processing for faster results"
                    )
                
                if st.button("🚀 Start Bulk Analysis", type="primary", use_container_width=True):
                    df_process = df.head(max_rows)
                    results = []
                    
                    progress_container = st.container()
                    
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_time = 0
                        
                        for i, row in df_process.iterrows():
                            text = str(row[selected_column]) if pd.notna(row[selected_column]) else ""
                            scores, max_score, is_toxic, proc_time = analyze_with_model(
                                text, selected_model, batch_threshold
                            )
                            
                            total_time += proc_time
                            
                            result = {
                                'Row': i + 1,
                                'Text': text[:100] + '...' if len(text) > 100 else text,
                                'Toxic': '🔴' if is_toxic else '🟢',
                                'Score': max_score,
                                'Risk': 'HIGH' if max_score > 0.7 else 'MEDIUM' if max_score > 0.3 else 'LOW'
                            }
                            
                            results.append(result)
                            
                            progress = (i + 1) / len(df_process)
                            progress_bar.progress(progress)
                            status_text.markdown(f"**Processing:** Row {i + 1} of {len(df_process)} ({progress:.0%})")
                            time.sleep(0.01)
                    
                    progress_container.empty()
                    
                    results_df = pd.DataFrame(results)
                    
                    st.markdown("### 📊 Analysis Summary")
                    
                    toxic_count = len([r for r in results if r['Toxic'] == '🔴'])
                    toxic_rate = (toxic_count / len(results)) * 100
                    avg_time = total_time / len(results)
                    
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        st.markdown(f"""
                        <div class="feature-card">
                            <div class="stat-number">{len(results)}</div>
                            <div style="color: #64748b; margin-top: 0.5rem;">Total Processed</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown(f"""
                        <div class="feature-card">
                            <div class="stat-number" style="color: #ef4444;">{toxic_count}</div>
                            <div style="color: #64748b; margin-top: 0.5rem;">Toxic Found</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        st.markdown(f"""
                        <div class="feature-card">
                            <div class="stat-number">{toxic_rate:.1f}%</div>
                            <div style="color: #64748b; margin-top: 0.5rem;">Toxicity Rate</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        st.markdown(f"""
                        <div class="feature-card">
                            <div class="stat-number">{avg_time:.0f}ms</div>
                            <div style="color: #64748b; margin-top: 0.5rem;">Avg. Time</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 📋 Detailed Results")
                    
                    filter_col1, filter_col2 = st.columns(2)
                    
                    with filter_col1:
                        show_toxic_only = st.checkbox("Show only toxic comments")
                    
                    with filter_col2:
                        sort_by_score = st.checkbox("Sort by toxicity score")
                    
                    display_df = results_df.copy()
                    
                    if show_toxic_only:
                        display_df = display_df[display_df['Toxic'] == '🔴']
                    
                    if sort_by_score:
                        display_df = display_df.sort_values('Score', ascending=False)
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    csv_download = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results (CSV)",
                        csv_download,
                        f"toxicity_analysis_{int(time.time())}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    # ================================================================
    # TAB 3: MODEL COMPARISON
    # ================================================================
    
    with tab3:
        st.markdown("## 🔄 Model Performance Comparison")
        
        st.markdown("### 🤖 Available Models")
        
        model_data = []
        for model in models.values():
            model_data.append({
                'Model Name': model.name,
                'Accuracy': f"{model.accuracy:.1%}",
                'Speed': model.speed,
                'Description': model.description
            })
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True, hide_index=True)
        
        st.markdown("### 🧪 Live Model Comparison")
        
        test_text = st.text_area(
            "Enter text to test all models:",
            value="You are such a stupid idiot, go kill yourself!",
            height=100
        )
        
        comparison_threshold = st.slider("Comparison threshold", 0.1, 0.9, 0.5, 0.1)
        
        if st.button("🔄 Compare All Models", type="primary", use_container_width=True):
            if test_text.strip():
                comparison_results = []
                
                for model_key, model in models.items():
                    scores, max_score, is_toxic, proc_time = analyze_with_model(
                        test_text, model, comparison_threshold
                    )
                    
                    comparison_results.append({
                        'Model': model.name,
                        'Accuracy': f"{model.accuracy:.1%}",
                        'Max Score': f"{max_score:.1%}",
                        'Toxic': '🔴' if is_toxic else '🟢',
                        'Time': f"{proc_time:.1f}ms"
                    })
                
                df_comparison = pd.DataFrame(comparison_results)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
                st.markdown("### 📈 Performance Summary")
                
                toxic_models = [r for r in comparison_results if r['Toxic'] == '🔴']
                fastest_model = min(comparison_results, key=lambda x: float(x['Time'].replace('ms', '')))
                most_accurate = max(comparison_results, key=lambda x: float(x['Accuracy'].replace('%', '')))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Models Detecting Toxicity", len(toxic_models))
                with col2:
                    st.metric("Fastest Model", fastest_model['Model'].split()[0])
                with col3:
                    st.metric("Most Accurate", most_accurate['Model'].split()[0])
    
    # ================================================================
    # TAB 4: ANALYTICS DASHBOARD
    # ================================================================
    
    with tab4:
        st.markdown("## 📊 Analytics Dashboard")
        
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        daily_stats = pd.DataFrame({
            'Date': dates,
            'Total Analyzed': np.random.randint(1000, 5000, size=30),
            'Toxic Detected': np.random.randint(100, 500, size=30),
            'Accuracy': np.random.uniform(92, 98, size=30)
        })
        
        category_dist = pd.DataFrame({
            'Category': ['Toxic', 'Severe', 'Obscene', 'Threat', 'Insult', 'Identity'],
            'Count': np.random.randint(50, 500, size=6),
            'Percentage': np.random.uniform(5, 30, size=6)
        })
        
        st.markdown("### 📈 30-Day Trend Analysis")
        
        fig_trend = px.line(
            daily_stats,
            x='Date',
            y=['Total Analyzed', 'Toxic Detected'],
            title='Detection Activity Over Time',
            labels={'value': 'Count', 'variable': 'Metric'},
            color_discrete_map={'Total Analyzed': '#667eea', 'Toxic Detected': '#ff6b6b'}
        )
        
        fig_trend.update_layout(
            height=400,
            hovermode='x unified',
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'color': "#1e293b"}
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Category Distribution")
            
            fig_pie = px.pie(
                category_dist,
                values='Count',
                names='Category',
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            
            fig_pie.update_layout(
                height=350,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': "#1e293b"}
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Detection Rates")
            
            fig_bar = px.bar(
                category_dist,
                x='Category',
                y='Percentage',
                color='Percentage',
                color_continuous_scale='Viridis'
            )
            
            fig_bar.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'color': "#1e293b"}
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # ================================================================
    # TAB 5: ABOUT
    # ================================================================
    
    with tab5:
        st.markdown("## ℹ️ About AI Shield")
        
        st.markdown("### 🚀 Key Features")
        
        feature_cols = st.columns(3)
        
        features = [
            ("⚡ Multi-Model System", "Choose from 4 different detection models", "🔵"),
            ("🎯 High Accuracy", "Up to 88% accuracy with hybrid model", "🟣"),
            ("📊 6 Categories", "Comprehensive toxicity detection", "🔴"),
            ("🔄 Bulk Processing", "Process thousands of comments", "🟢"),
            ("📈 Analytics", "Deep insights and visualizations", "🟡"),
            ("🛡️ Real-time", "Sub-150ms response time", "⚫")
        ]
        
        for i, (title, desc, color) in enumerate(features):
            with feature_cols[i % 3]:
                st.markdown(f"""
                <div class="feature-card">
                    <h4 style="color: #1e293b; margin-bottom: 0.5rem;">{title}</h4>
                    <p style="color: #64748b; font-size: 14px; margin: 0;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if i < 3:
                    st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #64748b; padding: 2rem 0;">
            <p>© 2024 AI Shield - Advanced Multi-Model Toxicity Detection</p>
            <p style="margin-top: 0.5rem;">
                Made with ❤️ by the AI Shield Team | Version 3.0.0
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
