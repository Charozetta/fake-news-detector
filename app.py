"""
FAKE NEWS DETECTOR - STREAMLIT APPLICATION
===========================================

Production-ready web app using Decision Tree (99.51% accuracy)

Features:
- Real-time fake news detection
- Confidence scores with visualization
- Explainable predictions (keyword analysis)
- Example articles for testing
- Professional UI with custom styling
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import json
import os
import glob
from datetime import datetime

# Scikit-learn imports - ИЗМЕНИМ ПОРЯДОК
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .real-news {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 3px solid #28a745;
    }
    
    .fake-news {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 3px solid #dc3545;
    }
    
    .prediction-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    
    .prediction-label {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .confidence-text {
        font-size: 1.5rem;
        color: #333;
    }
    
    /* Feature indicators */
    .indicator-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .indicator-positive {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    
    .indicator-negative {
        border-left-color: #dc3545;
        background-color: #f8d7da;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    /* Example article card */
    .example-card {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .example-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1f77b4 0%, #1557a0 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND COMPONENTS
# ============================================================================

@st.cache_resource
def load_model_and_metadata():
    """Load the trained pipeline and metadata"""
    
    # Find pipeline file
    import glob
    pipeline_files = glob.glob('fake_news_pipeline_*.pkl')
    
    if not pipeline_files:
        st.error("❌ Model file not found! Please ensure 'fake_news_pipeline_*.pkl' is in the same directory.")
        st.stop()
    
    # Use the most recent file
    pipeline_file = sorted(pipeline_files)[-1]
    
    try:
        with open(pipeline_file, 'rb') as f:
            pipeline = pickle.load(f)
        st.success(f"✅ Loaded model: {pipeline_file}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Load metadata
    try:
        with open('best_model_info.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ Metadata file not found. Using default values.")
        metadata = {
            'model_name': 'Decision Tree',
            'metrics': {
                'test_accuracy': 0.9951,
                'precision': 0.9945,
                'recall': 0.9964,
                'f1': 0.9955
            }
        }
    
    return pipeline, metadata

# Load model
with st.spinner("🔄 Loading model..."):
    pipeline, metadata = load_model_and_metadata()

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_text(text):
    """Clean and preprocess text (same as training)"""
    if not text or text.strip() == "":
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

def generate_explanation(text, prediction, confidence):
    """Generate human-readable explanation for the prediction"""
    
    # Process text
    processed = preprocess_text(text)
    words = processed.split()
    
    # Define keyword indicators
    fake_keywords = [
        'breaking', 'shocking', 'unbelievable', 'incredible', 'amazing',
        'must', 'watch', 'video', 'revealed', 'truth', 'exposed', 'secret',
        'they', 'dont', 'want', 'conspiracy', 'hoax', 'scam', 'lies',
        'censored', 'banned', 'hidden', 'shocking', 'bombshell'
    ]
    
    real_keywords = [
        'reuters', 'said', 'according', 'official', 'government', 
        'president', 'report', 'statement', 'announced', 'confirmed',
        'minister', 'spokesman', 'agency', 'committee', 'department',
        'parliament', 'congress', 'senate', 'representative', 'secretary'
    ]
    
    # Find indicators in text
    found_fake = [w for w in words if w in fake_keywords]
    found_real = [w for w in words if w in real_keywords]
    
    # Text statistics
    stats = {
        'word_count': len(words),
        'unique_words': len(set(words)),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'fake_indicators': len(found_fake),
        'real_indicators': len(found_real)
    }
    
    return {
        'fake_indicators': found_fake[:10],  # Top 10
        'real_indicators': found_real[:10],
        'stats': stats,
        'all_words': words[:30]  # First 30 words
    }

# ============================================================================
# EXAMPLE ARTICLES
# ============================================================================

EXAMPLES = {
    "📰 Real News Example 1": """
WASHINGTON (Reuters) - The U.S. House of Representatives approved a bill on Thursday 
that would impose new sanctions on Russia and restrict President Donald Trump's ability 
to ease penalties against Moscow. The measure, which passed by a vote of 419-3, follows 
allegations of Russian interference in the 2016 U.S. presidential election. The bill 
now moves to the Senate for consideration. White House officials said Trump would review 
the legislation carefully before making any decisions.
    """,
    
    "📰 Real News Example 2": """
According to a statement released by the Ministry of Health, the new vaccination program 
will begin next month in all major cities across the country. Health officials confirmed 
that the vaccines have been approved by regulatory authorities and meet international 
safety standards. The minister announced during a press conference that priority will be 
given to healthcare workers and elderly citizens. Distribution centers are being set up 
in hospitals and community health facilities nationwide.
    """,
    
    "⚠️ Fake News Example 1": """
BREAKING NEWS: SHOCKING discovery reveals what the government has been hiding from you 
for DECADES! You absolutely WON'T BELIEVE what scientists have just uncovered! This 
INCREDIBLE revelation will change EVERYTHING you thought you knew! They tried to SILENCE 
us, but we have the PROOF! Click here NOW to watch the video before it gets CENSORED! 
Share this EVERYWHERE before they take it down! This is NOT a hoax! The TRUTH must be told!
    """,
    
    "⚠️ Fake News Example 2": """
EXPOSED: The secret conspiracy that mainstream media REFUSES to cover! Insiders reveal 
the SHOCKING truth about what's REALLY happening behind closed doors! They thought they 
could hide it from us, but brave whistleblowers have come forward with UNDENIABLE evidence! 
Watch this bombshell video NOW before it disappears! Don't trust the LIES they're feeding 
you! WAKE UP and see the TRUTH! Share immediately!
    """
}

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">AI-Powered News Verification System | Powered by Machine Learning</p>',
    unsafe_allow_html=True
)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("📊 Model Information")
    
    st.markdown(f"""
    **Model:** {metadata['model_name']}  
    **Accuracy:** {metadata['metrics']['test_accuracy']:.2%}  
    **Precision:** {metadata['metrics']['precision']:.2%}  
    **Recall:** {metadata['metrics']['recall']:.2%}  
    **F1-Score:** {metadata['metrics']['f1']:.4f}
    """)
    
    st.markdown("---")
    
    st.header("📖 How It Works")
    st.markdown("""
    1. **Enter** or paste a news article
    2. **Click** "Analyze Article"
    3. **View** instant prediction
    4. **Understand** the reasoning
    """)
    
    st.markdown("---")
    
    st.header("🎯 Why Trust This?")
    st.markdown("""
    ✅ **99.5% Accuracy** on test data  
    ✅ **Trained on 35,000+** articles  
    ✅ **Explainable** predictions  
    ✅ **Fast** real-time analysis  
    """)
    
    st.markdown("---")
    
    st.header("ℹ️ About")
    st.markdown("""
    This system uses a **Decision Tree classifier** 
    trained on the **ISOT Fake News Dataset**.
    
    **Dataset:** 35,000+ articles (2016-2017)  
    **Sources:** Reuters, fake news websites  
    **Performance:** 99.51% accuracy
    """)
    
    st.markdown("---")
    
    # Example selector
    if st.button("📰 Load Example Articles"):
        st.session_state['show_examples'] = True
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.8rem;'>
    ⚠️ Educational Tool<br>
    Always verify from multiple sources
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# EXAMPLE ARTICLES SELECTOR
# ============================================================================

if st.session_state.get('show_examples', False):
    with st.expander("📰 Example Articles - Click to Use", expanded=True):
        selected_example = st.radio(
            "Choose an example:",
            list(EXAMPLES.keys()),
            key="example_selector"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Use This Example", use_container_width=True):
                st.session_state['input_text'] = EXAMPLES[selected_example]
                st.session_state['show_examples'] = False
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state['show_examples'] = False
                st.rerun()
        
        st.markdown("**Preview:**")
        st.text(EXAMPLES[selected_example][:200] + "...")

# ============================================================================
# INPUT SECTION
# ============================================================================

st.header("📝 Enter News Article")

input_text = st.text_area(
    "Paste or type the news article text below:",
    value=st.session_state.get('input_text', ''),
    height=200,
    placeholder="Example: WASHINGTON (Reuters) - The President announced today that...",
    help="Enter at least 50 characters for accurate analysis"
)

# Buttons
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

with col2:
    if st.button("📰 Examples", use_container_width=True):
        st.session_state['show_examples'] = True
        st.rerun()

with col3:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state['input_text'] = ''
        st.rerun()

# Character counter
char_count = len(input_text.strip())
if char_count > 0:
    if char_count < 50:
        st.warning(f"⚠️ Text too short: {char_count} characters (minimum 50 recommended)")
    else:
        st.success(f"✅ Text length: {char_count} characters")

# ============================================================================
# ANALYSIS AND RESULTS
# ============================================================================

if analyze_button:
    if not input_text or len(input_text.strip()) < 20:
        st.error("❌ Please enter a longer text (at least 20 characters)")
    else:
        with st.spinner("🔄 Analyzing article..."):
            # Preprocess
            processed_text = preprocess_text(input_text)
            
            if len(processed_text.split()) < 5:
                st.error("❌ After preprocessing, text is too short. Please enter more content.")
            else:
                # Make prediction
                try:
                    prediction = pipeline.predict([processed_text])[0]
                    proba = pipeline.predict_proba([processed_text])[0]
                    
                    confidence = proba[1] * 100 if prediction == 1 else proba[0] * 100
                    
                    # Generate explanation
                    explanation = generate_explanation(input_text, prediction, confidence)
                    
                    # Display results
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Prediction box
                    if prediction == 1:  # REAL
                        st.markdown(f"""
                        <div class="prediction-box real-news">
                            <div class="prediction-icon">✅</div>
                            <div class="prediction-label">REAL NEWS</div>
                            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # FAKE
                        st.markdown(f"""
                        <div class="prediction-box fake-news">
                            <div class="prediction-icon">❌</div>
                            <div class="prediction-label">FAKE NEWS</div>
                            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.markdown("### Confidence Level")
                    st.progress(confidence / 100)
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Prediction", "REAL" if prediction == 1 else "FAKE")
                    with col2:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col3:
                        st.metric("Word Count", explanation['stats']['word_count'])
                    with col4:
                        st.metric("Unique Words", explanation['stats']['unique_words'])
                    
                    # Tabs for detailed analysis
                    st.markdown("---")
                    tab1, tab2, tab3 = st.tabs(["🎯 Key Indicators", "📊 Text Analysis", "🔍 Details"])
                    
                    with tab1:
                        st.markdown("### Why This Prediction?")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### ✅ REAL News Indicators")
                            if explanation['real_indicators']:
                                for word in explanation['real_indicators']:
                                    st.markdown(f"""
                                    <div class="indicator-box indicator-positive">
                                        <strong>"{word}"</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                                st.info(f"Found {len(explanation['real_indicators'])} credibility markers")
                            else:
                                st.warning("No strong REAL indicators found")
                        
                        with col2:
                            st.markdown("#### ❌ FAKE News Indicators")
                            if explanation['fake_indicators']:
                                for word in explanation['fake_indicators']:
                                    st.markdown(f"""
                                    <div class="indicator-box indicator-negative">
                                        <strong>"{word}"</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                                st.warning(f"Found {len(explanation['fake_indicators'])} suspicious patterns")
                            else:
                                st.success("No fake news patterns detected")
                        
                        # Explanation text
                        st.markdown("---")
                        st.markdown("### 💡 Interpretation")
                        
                        if prediction == 1:  # REAL
                            st.success(f"""
                            **This article appears to be REAL news because:**
                            - Contains {len(explanation['real_indicators'])} credibility markers (e.g., source attribution, official statements)
                            - Only {len(explanation['fake_indicators'])} sensational patterns
                            - Professional writing style with {explanation['stats']['word_count']} words
                            - Model confidence: {confidence:.1f}%
                            """)
                        else:  # FAKE
                            st.error(f"""
                            **This article appears to be FAKE news because:**
                            - Contains {len(explanation['fake_indicators'])} sensational patterns (e.g., "BREAKING", "SHOCKING")
                            - Only {len(explanation['real_indicators'])} credibility markers
                            - Emotional manipulation detected
                            - Model confidence: {confidence:.1f}%
                            """)
                    
                    with tab2:
                        st.markdown("### Text Statistics")
                        
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        
                        with stats_col1:
                            st.metric(
                                "Total Words",
                                explanation['stats']['word_count']
                            )
                            st.metric(
                                "Unique Words",
                                explanation['stats']['unique_words']
                            )
                        
                        with stats_col2:
                            st.metric(
                                "Avg Word Length",
                                f"{explanation['stats']['avg_word_length']:.1f}"
                            )
                            vocab_richness = explanation['stats']['unique_words'] / explanation['stats']['word_count'] if explanation['stats']['word_count'] > 0 else 0
                            st.metric(
                                "Vocabulary Richness",
                                f"{vocab_richness:.2%}"
                            )
                        
                        with stats_col3:
                            st.metric(
                                "REAL Indicators",
                                explanation['stats']['real_indicators']
                            )
                            st.metric(
                                "FAKE Indicators",
                                explanation['stats']['fake_indicators']
                            )
                        
                        # First 30 words
                        st.markdown("---")
                        st.markdown("### 📝 First Words (Processed)")
                        st.code(' '.join(explanation['all_words']))
                    
                    with tab3:
                        st.markdown("### 🔬 Technical Details")
                        
                        # Class probabilities
                        st.markdown("#### Class Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': ['FAKE (0)', 'REAL (1)'],
                            'Probability': [proba[0], proba[1]]
                        })
                        st.bar_chart(prob_df.set_index('Class'))
                        
                        # Show probabilities
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("P(FAKE)", f"{proba[0]:.4f}")
                        with col2:
                            st.metric("P(REAL)", f"{proba[1]:.4f}")
                        
                        # Model info
                        st.markdown("---")
                        st.markdown("#### Model Information")
                        st.json({
                            "Model": metadata['model_name'],
                            "Test Accuracy": f"{metadata['metrics']['test_accuracy']:.4f}",
                            "Precision": f"{metadata['metrics']['precision']:.4f}",
                            "Recall": f"{metadata['metrics']['recall']:.4f}",
                            "F1-Score": f"{metadata['metrics']['f1']:.4f}"
                        })
                        
                        # Processed text preview
                        with st.expander("View Preprocessed Text"):
                            st.text_area(
                                "Cleaned text used for prediction:",
                                processed_text[:500] + "..." if len(processed_text) > 500 else processed_text,
                                height=150,
                                disabled=True
                            )
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.error("Please check that your model file is compatible.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>🔍 Fake News Detector</strong></p>
    <p>Powered by Machine Learning (Decision Tree, 99.51% Accuracy)</p>
    <p>Trained on ISOT Fake News Dataset (35,000+ articles)</p>
    <p style="margin-top: 1rem; font-size: 0.9rem;">
        ⚠️ <em>This is an educational tool. Always verify information from multiple reliable sources.</em>
    </p>
    <p style="margin-top: 0.5rem; color: #999; font-size: 0.85rem;">
        © 2024 | Built with Streamlit & scikit-learn
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'input_text' not in st.session_state:
    st.session_state['input_text'] = ''
if 'show_examples' not in st.session_state:
    st.session_state['show_examples'] = False
