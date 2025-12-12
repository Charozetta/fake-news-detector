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
import re
import json
import os
import glob
from datetime import datetime
import joblib

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

CUSTOM_CSS = """
<style>
/* Global styles */
main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Title styling */
.title-container {
    text-align: left;
    margin-bottom: 1.5rem;
}

.title-badge {
    display: inline-block;
    background: linear-gradient(90deg, #ff4b4b, #ff9f43);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.title-main {
    font-size: 2.4rem;
    font-weight: 800;
    margin-top: 0.5rem;
    margin-bottom: 0.4rem;
    letter-spacing: 0.02em;
}

.title-subtitle {
    color: #6c757d;
    font-size: 0.98rem;
    max-width: 600px;
}

/* Layout cards */
.stAlert {
    border-radius: 12px;
}

/* Metric cards */
.metric-card {
    padding: 0.75rem 1rem;
    border-radius: 10px;
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    margin-bottom: 0.5rem;
}

.metric-title {
    font-size: 0.85rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.1rem;
}

.metric-value {
    font-size: 1.2rem;
    font-weight: 700;
}

/* Text area styling */
textarea {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Prediction badge */
.prediction-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 600;
}

/* Fake / Real colors */
.fake-badge {
    background: rgba(220, 53, 69, 0.08);
    color: #dc3545;
    border: 1px solid rgba(220, 53, 69, 0.4);
}

.real-badge {
    background: rgba(40, 167, 69, 0.08);
    color: #28a745;
    border: 1px solid rgba(40, 167, 69, 0.4);
}

/* Confidence bar container */
.confidence-bar-container {
    width: 100%;
    background-color: #f1f3f5;
    border-radius: 999px;
    overflow: hidden;
    height: 20px;
    margin: 0.4rem 0 0.2rem 0;
    border: 1px solid #dee2e6;
}

/* Confidence bar fill */
.confidence-bar-fill-fake {
    height: 100%;
    background: linear-gradient(90deg, #ff6b6b 0%, #dc3545 100%);
}

.confidence-bar-fill-real {
    height: 100%;
    background: linear-gradient(90deg, #51cf66 0%, #28a745 100%);
}

/* Confidence label */
.confidence-label {
    font-size: 0.85rem;
    color: #495057;
}

/* Explanation list */
.explanation-list {
    margin-top: 0.3rem;
}

.explanation-item {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    background-color: #f1f3f5;
    margin: 0.08rem;
    font-size: 0.8rem;
}

/* Sidebar */
.css-1d391kg, .css-1d391kg.e1fqkh3o3 {  /* sidebar container */
    padding-top: 1.5rem;
}

/* Footer */
.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #e9ecef;
    color: #868e96;
    font-size: 0.85rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# MODEL & METADATA LOADING
# ============================================================================


def load_model_and_metadata():
    """Load the trained pipeline and metadata"""

    # Find pipeline file
    pipeline_files = glob.glob("fake_news_pipeline_*.pkl")

    if not pipeline_files:
        st.error(
            "❌ Model file not found! Please ensure 'fake_news_pipeline_*.pkl' "
            "is in the same directory."
        )
        st.stop()

    # Use the most recent file (по имени, если есть timestamp)
    pipeline_file = sorted(pipeline_files)[-1]

    # Load model with joblib (тем же способом, как она была сохранена)
    try:
        pipeline = joblib.load(pipeline_file)
        st.success(f"✅ Loaded model: {pipeline_file}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # Load metadata if available
    metadata_file = "best_model_info.json"
    metadata = None

    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(
                f"⚠️ Could not load metadata from {metadata_file}: {e}"
            )
            metadata = {}
    else:
        st.warning("⚠️ Metadata file not found. Using default values.")
        metadata = {
            "model_name": "Decision Tree",
            "metrics": {
                "test_accuracy": 0.9951,
                "precision": 0.9945,
                "recall": 0.9964,
                "f1": 0.9955,
            },
        }

    return pipeline, metadata


# Load model
with st.spinner("🔄 Loading model..."):
    pipeline, metadata = load_model_and_metadata()

# ============================================================================
# TEXT PREPROCESSING (same as in training)
# ============================================================================


def preprocess_text(text: str) -> str:
    """Preprocess text similar to training pipeline"""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================


def generate_explanation(text, prediction, confidence):
    """Generate human-readable explanation for the prediction"""

    # Process text
    processed = preprocess_text(text)
    words = processed.split()

    # Very simple keyword-based explanation for demo purposes
    fake_keywords = [
        "shocking",
        "breaking",
        "exclusive",
        "you won",
        "click",
        "share",
        "viral",
        "conspiracy",
        "secret",
        "rumor",
    ]
    real_keywords = [
        "according",
        "report",
        "official",
        "data",
        "study",
        "research",
        "analysis",
        "statement",
        "confirmed",
        "source",
    ]

    used_fake_keywords = [
        kw for kw in fake_keywords if kw in processed.lower()
    ]
    used_real_keywords = [
        kw for kw in real_keywords if kw in processed.lower()
    ]

    explanation = {
        "fake_keywords": used_fake_keywords,
        "real_keywords": used_real_keywords,
    }

    return explanation


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Model Information")

    model_name = metadata.get("model_name", "Decision Tree")
    metrics = metadata.get("metrics", {})

    st.markdown(
        f"**Model:** `{model_name}`  \n"
        f"**Vectorizer:** `{metadata.get('vectorizer_name', 'TF-IDF')}`"
    )

    st.markdown("---")
    st.markdown("#### 📊 Key metrics")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        acc = metrics.get("test_accuracy", 0.0)
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Accuracy</div>"
            f"<div class='metric-value'>{acc * 100:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        prec = metrics.get("precision", 0.0)
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Precision</div>"
            f"<div class='metric-value'>{prec * 100:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_m2:
        rec = metrics.get("recall", 0.0)
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Recall</div>"
            f"<div class='metric-value'>{rec * 100:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        f1 = metrics.get("f1", 0.0)
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>F1-score</div>"
            f"<div class='metric-value'>{f1 * 100:.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 🧪 Try example texts")

    example_fake = st.checkbox("Fill with example *FAKE* article")
    example_real = st.checkbox("Fill with example *REAL* article")

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Title section
st.markdown(
    """
<div class="title-container">
    <div class="title-badge">Fake News Detection • Demo</div>
    <h1 class="title-main">Real-time Fake News Classifier</h1>
    <p class="title-subtitle">
        Paste any news article or statement, and the model will predict whether it is likely 
        to be <strong>fake</strong> or <strong>real</strong>, along with confidence and simple explanation.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Layout columns
left_col, right_col = st.columns([1.4, 1])

# ============================================================================
# LEFT COLUMN - INPUT
# ============================================================================

with left_col:
    st.markdown("#### 📝 Input text")

    default_text = ""

    if example_fake:
        default_text = (
            "BREAKING: Scientists discover a secret cure for all diseases, "
            "but pharmaceutical companies are hiding it to keep profits high."
        )
    elif example_real:
        default_text = (
            "According to a recent report published by the World Health "
            "Organization, vaccination campaigns have significantly "
            "reduced the incidence of measles worldwide."
        )

    text_input = st.text_area(
        "Paste news article or statement here:",
        value=default_text,
        height=260,
        placeholder="Paste or type your news text here...",
    )

    analyze_button = st.button("🔍 Analyze Text", type="primary")

# ============================================================================
# RIGHT COLUMN - OUTPUT
# ============================================================================

with right_col:
    st.markdown("#### 📈 Prediction results")

    if analyze_button:
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Predict using the pipeline directly
                prediction = pipeline.predict([text_input])[0]

                # Try to get probabilities if supported
                fake_confidence = None
                real_confidence = None

                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba([text_input])[0]
                    # Assuming binary classification [FAKE, REAL]
                    fake_confidence = float(proba[0])
                    real_confidence = float(proba[1])
                else:
                    # If no probability, we just use 0.5 vs 0.5 or 0.9 vs 0.1 as a heuristic
                    if prediction == "FAKE":
                        fake_confidence = 0.9
                        real_confidence = 0.1
                    else:
                        fake_confidence = 0.1
                        real_confidence = 0.9

                # Determine label and styles
                is_fake = prediction == "FAKE"
                main_label = "FAKE news" if is_fake else "REAL news"
                confidence = fake_confidence if is_fake else real_confidence

                badge_class = "fake-badge" if is_fake else "real-badge"
                emoji = "🚨" if is_fake else "✅"

                # Prediction badge
                st.markdown(
                    f"""
                    <div class="prediction-badge {badge_class}">
                        <span>{emoji}</span>
                        <span>{main_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Confidence bar
                conf_percent = confidence * 100 if confidence is not None else 0

                if is_fake:
                    fill_class = "confidence-bar-fill-fake"
                else:
                    fill_class = "confidence-bar-fill-real"

                st.markdown(
                    f"""
                    <div class="confidence-label">
                        Model confidence: <strong>{conf_percent:.1f}%</strong>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="{fill_class}" style="width: {conf_percent:.1f}%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Explanation
                explanation = generate_explanation(
                    text_input, prediction, confidence
                )

                st.markdown("##### 🔍 Explanation (keywords)")

                if (
                    not explanation["fake_keywords"]
                    and not explanation["real_keywords"]
                ):
                    st.write(
                        "No specific indicative keywords found in the text. "
                        "The model relied on overall patterns in the text."
                    )
                else:
                    if explanation["fake_keywords"]:
                        st.markdown("**Fake-related keywords found:**")
                        fake_kw_html = "".join(
                            f"<span class='explanation-item'>{kw}</span>"
                            for kw in explanation["fake_keywords"]
                        )
                        st.markdown(
                            f"<div class='explanation-list'>{fake_kw_html}</div>",
                            unsafe_allow_html=True,
                        )

                    if explanation["real_keywords"]:
                        st.markdown("**Real-related keywords found:**")
                        real_kw_html = "".join(
                            f"<span class='explanation-item'>{kw}</span>"
                            for kw in explanation["real_keywords"]
                        )
                        st.markdown(
                            f"<div class='explanation-list'>{real_kw_html}</div>",
                            unsafe_allow_html=True,
                        )

    else:
        st.info(
            "Enter a news text on the left and click **Analyze Text** "
            "to see prediction and explanation here."
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    """
<div class="footer">
    <p>
        This demo is based on a Decision Tree model trained on the ISOT Fake News dataset.
        It is for educational purposes only and should not be used as a sole source
        of truth for critical decisions.
    </p>
    <p style="margin-top: 0.5rem; color: #999; font-size: 0.85rem;">
        © 2024 | Built with Streamlit & scikit-learn
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""
if "show_examples" not in st.session_state:
    st.session_state["show_examples"] = False
