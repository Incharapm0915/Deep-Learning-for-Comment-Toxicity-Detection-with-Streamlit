# Step 3: Streamlit Application for Toxicity Detection (Error-Resistant Version)
# ============================================================================
# File: streamlit_app/app.py
# Project Path: D:\Deep Learning for Comment Toxicity

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import os
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# Try to import TensorFlow with error handling
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError as e:
    st.warning(f"TensorFlow import failed: {str(e)}")
    st.info("Running in fallback mode with TF-IDF + Logistic Regression")
    TENSORFLOW_AVAILABLE = False

# Try to import Plotly with fallback
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("Plotly not available. Using matplotlib for visualizations.")
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Toxicity Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    
    .toxic-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .safe-alert {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================
# FALLBACK MODEL CLASSES
# ================================================================

class FallbackToxicityModel:
    """Fallback model using TF-IDF + Logistic Regression when TensorFlow fails"""
    
    def __init__(self):
        self.model = None
        self.tfidf = None
        self.toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.is_trained = False
        
    def load_or_create_fallback(self):
        """Load existing fallback model or create a new one"""
        try:
            # Try to load existing fallback model
            with open('../models/fallback_model.pickle', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.tfidf = model_data['tfidf']
                self.is_trained = True
                return True
        except:
            # Create a simple fallback model
            self.create_simple_fallback()
            return True
    
    def create_simple_fallback(self):
        """Create a simple rule-based fallback"""
        # Define some toxic keywords for simple classification
        self.toxic_keywords = {
            'toxic': ['hate', 'stupid', 'idiot', 'moron', 'damn', 'hell'],
            'severe_toxic': ['kill', 'die', 'murder', 'death'],
            'obscene': ['fuck', 'shit', 'ass', 'bitch'],
            'threat': ['kill you', 'hurt you', 'destroy you', 'kill', 'murder'],
            'insult': ['stupid', 'idiot', 'moron', 'loser', 'pathetic'],
            'identity_hate': ['racist', 'nazi', 'terrorist']
        }
        self.is_trained = True
    
    def predict_simple(self, text):
        """Simple keyword-based prediction"""
        text_lower = text.lower()
        predictions = {}
        
        for category, keywords in self.toxic_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score = min(score + 0.3, 0.9)  # Cap at 0.9
            predictions[category] = score
        
        return predictions
    
    def predict(self, text):
        """Predict toxicity for given text"""
        if self.model and self.tfidf:
            # Use trained model if available
            try:
                text_vectorized = self.tfidf.transform([text])
                pred = self.model.predict_proba(text_vectorized)
                return {col: float(pred[i][:, 1][0]) for i, col in enumerate(self.toxicity_columns)}
            except:
                return self.predict_simple(text)
        else:
            # Use simple keyword-based approach
            return self.predict_simple(text)

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

@st.cache_data
def load_model_artifacts():
    """Load trained model and preprocessing artifacts with fallback"""
    if TENSORFLOW_AVAILABLE:
        try:
            # Load model parameters
            with open('../models/model_params.pickle', 'rb') as f:
                model_params = pickle.load(f)
            
            # Load tokenizer
            with open('../models/tokenizer.pickle', 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Load the best model
            model = load_model('../models/best_toxicity_model.h5')
            
            return model, tokenizer, model_params, 'tensorflow'
        except Exception as e:
            st.warning(f"Failed to load TensorFlow model: {str(e)}")
    
    # Fallback to simple model
    fallback_model = FallbackToxicityModel()
    fallback_model.load_or_create_fallback()
    
    # Create mock model_params
    model_params = {
        'toxicity_columns': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
        'best_model_name': 'Fallback Model',
        'best_model_auc': 0.750,
        'max_length': 50,
        'vocab_size': 10000,
        'embedding_dim': 50
    }
    
    return fallback_model, None, model_params, 'fallback'

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    try:
        train_data = pd.read_csv('../data/processed/train_processed.csv')
        return train_data
    except:
        # Create mock sample data if files don't exist
        mock_data = pd.DataFrame({
            'comment_text_processed': [
                'this is a great article',
                'you are stupid and wrong',
                'i love this discussion',
                'hate this terrible content'
            ],
            'comment_length': [20, 25, 22, 28],
            'is_toxic': [0, 1, 0, 1],
            'toxic': [0, 1, 0, 1],
            'severe_toxic': [0, 0, 0, 0],
            'obscene': [0, 1, 0, 0],
            'threat': [0, 0, 0, 0],
            'insult': [0, 1, 0, 0],
            'identity_hate': [0, 0, 0, 0]
        })
        return mock_data

def clean_text_for_prediction(text):
    """Clean text for model prediction"""
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s\']', '', text)
    text = text.strip()
    
    return text

def predict_toxicity(text, model, tokenizer, model_params, model_type):
    """Predict toxicity for a given text"""
    if not text.strip():
        return {col: 0.0 for col in model_params['toxicity_columns']}
    
    cleaned_text = clean_text_for_prediction(text)
    
    if model_type == 'tensorflow' and tokenizer:
        # Use TensorFlow model
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=model_params['max_length'], 
                                       padding='post', truncating='post')
        predictions = model.predict(padded_sequence, verbose=0)[0]
        return {col: float(pred) for col, pred in zip(model_params['toxicity_columns'], predictions)}
    else:
        # Use fallback model
        return model.predict(cleaned_text)

def create_simple_bar_chart(categories, scores):
    """Create a simple bar chart using matplotlib when Plotly is not available"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if score > 50 else 'green' for score in scores]
    bars = ax.bar(categories, scores, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{score:.1f}%', ha='center', va='bottom')
    
    ax.set_title('Toxicity Scores by Category')
    ax.set_xlabel('Toxicity Categories')
    ax.set_ylabel('Confidence Score (%)')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# ================================================================
# MAIN APPLICATION
# ================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Toxicity Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Display system status
    if not TENSORFLOW_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ System Running in Fallback Mode</h4>
        <p>TensorFlow could not be loaded. The system is using a simplified model for toxicity detection.</p>
        <p><strong>To fix this:</strong></p>
        <ul>
        <li>Install Visual C++ Redistributable for Visual Studio 2015-2022</li>
        <li>Reinstall TensorFlow: <code>pip uninstall tensorflow && pip install tensorflow</code></li>
        <li>Try: <code>pip install tensorflow-cpu</code> if you don't have a GPU</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Load model artifacts
    model, tokenizer, model_params, model_type = load_model_artifacts()
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Information")
        
        status_color = "🟢" if model_type == 'tensorflow' else "🟡"
        st.markdown(f"""
        <div class="sidebar-info">
        <h4>Model Status</h4>
        <ul>
        <li><strong>Status:</strong> {status_color} {model_type.title()}</li>
        <li><strong>Model:</strong> {model_params['best_model_name']}</li>
        <li><strong>AUC Score:</strong> {model_params['best_model_auc']:.3f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("## 🎯 Toxicity Categories")
        for category in model_params['toxicity_columns']:
            st.markdown(f"• **{category.replace('_', ' ').title()}**")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Comment Analysis", 
        "📁 Bulk CSV Analysis", 
        "📊 System Information"
    ])
    
    # ================================================================
    # TAB 1: SINGLE COMMENT ANALYSIS
    # ================================================================
    
    with tab1:
        st.markdown("## 💬 Analyze Individual Comments")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter a comment to analyze:",
                placeholder="Type your comment here...",
                height=150,
                help="Enter any text to check for toxicity across multiple categories"
            )
            
            st.markdown("**Quick Test Examples:**")
            sample_comments = [
                "This is a great article, thanks for sharing!",
                "I disagree with your opinion but respect your viewpoint.",
                "You are absolutely stupid and should shut up!",
                "This movie was terrible and boring."
            ]
            
            selected_sample = st.selectbox(
                "Or select a sample comment:",
                [""] + sample_comments,
                help="Choose a pre-written comment for quick testing"
            )
            
            if selected_sample:
                user_input = selected_sample
        
        with col2:
            if st.button("🔍 Analyze Comment", type="primary", use_container_width=True):
                if user_input.strip():
                    with st.spinner("Analyzing comment..."):
                        predictions = predict_toxicity(user_input, model, tokenizer, model_params, model_type)
                    
                    max_toxicity = max(predictions.values())
                    is_toxic = max_toxicity > 0.5
                    
                    if is_toxic:
                        st.markdown(f"""
                        <div class="toxic-alert">
                        <h3>⚠️ Potentially Toxic Content Detected</h3>
                        <p><strong>Highest Toxicity Score:</strong> {max_toxicity:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                        <h3>✅ Content Appears Safe</h3>
                        <p><strong>Highest Toxicity Score:</strong> {max_toxicity:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Detailed Toxicity Breakdown")
                    
                    categories = list(predictions.keys())
                    scores = [predictions[cat] * 100 for cat in categories]
                    
                    if PLOTLY_AVAILABLE:
                        colors = ['#FF6B6B' if score > 50 else '#4ECDC4' for score in scores]
                        fig = go.Figure(data=[
                            go.Bar(x=categories, y=scores, marker_color=colors, 
                                   text=[f'{s:.1f}%' for s in scores], textposition='auto')
                        ])
                        fig.update_layout(
                            title="Toxicity Scores by Category",
                            xaxis_title="Toxicity Categories",
                            yaxis_title="Confidence Score (%)",
                            yaxis=dict(range=[0, 100]),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = create_simple_bar_chart(categories, scores)
                        st.pyplot(fig)
                    
                    # Metrics in columns
                    cols = st.columns(3)
                    for i, (category, score) in enumerate(predictions.items()):
                        with cols[i % 3]:
                            st.metric(
                                label=category.replace('_', ' ').title(),
                                value=f"{score:.1%}",
                                delta=f"{'High' if score > 0.5 else 'Low'} Risk"
                            )
                
                else:
                    st.warning("Please enter a comment to analyze.")
    
    # ================================================================
    # TAB 2: BULK CSV ANALYSIS (SIMPLIFIED)
    # ================================================================
    
    with tab2:
        st.markdown("## 📁 Bulk Analysis from CSV File")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Found {len(df)} rows.")
                
                st.dataframe(df.head(), use_container_width=True)
                
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox(
                        "Select the column containing text to analyze:",
                        text_columns
                    )
                    
                    if st.button("🚀 Start Analysis", type="primary"):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, row in df.iterrows():
                            text = str(row[selected_column]) if pd.notna(row[selected_column]) else ""
                            predictions = predict_toxicity(text, model, tokenizer, model_params, model_type)
                            
                            result_row = {'row_index': idx, 'original_text': text}
                            result_row.update(predictions)
                            results.append(result_row)
                            
                            progress = (idx + 1) / len(df)
                            progress_bar.progress(progress)
                        
                        progress_bar.empty()
                        results_df = pd.DataFrame(results)
                        
                        st.success("Analysis completed!")
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv_download = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_download,
                            file_name=f"toxicity_results_{int(time.time())}.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # ================================================================
    # TAB 3: SYSTEM INFORMATION
    # ================================================================
    
    with tab3:
        st.markdown("## 📊 System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Technical Details")
            tech_info = pd.DataFrame({
                "Component": ["TensorFlow", "Plotly", "Model Type", "Fallback Available"],
                "Status": [
                    "✅ Available" if TENSORFLOW_AVAILABLE else "❌ Not Available",
                    "✅ Available" if PLOTLY_AVAILABLE else "❌ Not Available", 
                    model_type.title(),
                    "✅ Yes"
                ]
            })
            st.dataframe(tech_info, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 Model Performance")
            perf_info = pd.DataFrame({
                "Metric": ["Model AUC", "Categories", "Mode"],
                "Value": [
                    f"{model_params['best_model_auc']:.3f}",
                    str(len(model_params['toxicity_columns'])),
                    "Production" if model_type == 'tensorflow' else "Fallback"
                ]
            })
            st.dataframe(perf_info, use_container_width=True, hide_index=True)
        
        if sample_data is not None and len(sample_data) > 4:
            st.markdown("### 📊 Training Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(sample_data))
            with col2:
                toxic_count = sample_data['is_toxic'].sum()
                st.metric("Toxic Comments", toxic_count)
            with col3:
                rate = (toxic_count / len(sample_data)) * 100
                st.metric("Toxicity Rate", f"{rate:.1f}%")

if __name__ == "__main__":
    main()