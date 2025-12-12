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
    if not input_text or len(input_text.strip()) < 50:
        st.error("❌ Please enter a longer text (at least 50 characters)")
    else:
        with st.spinner("🔄 Analyzing article..."):
            # 1. ПОДГОТАВЛИВАЕМ ТЕКСТ ТАК ЖЕ, КАК В НОУТБУКЕ
            processed_text = preprocess_text(input_text)

            if len(processed_text.split()) < 5:
                st.error("❌ After preprocessing, text is too short. Please enter more content.")
            else:
                # 2. ПРЕДСКАЗАНИЕ МОДЕЛИ
                raw_pred = pipeline.predict([processed_text])[0]

                # На всякий случай приводим к нормальному виду
                is_fake = False
                label_str = None

                # Числовые метки (0 = FAKE, 1 = REAL)
                import numpy as np
                if isinstance(raw_pred, (int, np.integer)):
                    is_fake = int(raw_pred) == 0
                    label_str = "FAKE" if is_fake else "REAL"

                # Строковые метки ("FAKE"/"REAL" и т.п.)
                elif isinstance(raw_pred, str):
                    lower = raw_pred.lower().strip()
                    if "fake" in lower or lower == "0":
                        is_fake = True
                        label_str = "FAKE"
                    else:
                        is_fake = False
                        label_str = "REAL"
                else:
                    # Фоллбэк — считаем всё REAL, но честно пишем тип
                    is_fake = False
                    label_str = f"{raw_pred}"

                # 3. ПРОБЫ (если есть)
                fake_confidence = None
                real_confidence = None
                confidence = 90.0  # дефолт на всякий случай

                if hasattr(pipeline, "predict_proba"):
                    try:
                        proba = pipeline.predict_proba([processed_text])[0]

                        # классы из модели
                        classes = list(pipeline.classes_)

                        # пытаемся найти индексы классов для FAKE и REAL
                        fake_idx = None
                        real_idx = None

                        # числовые метки
                        if isinstance(classes[0], (int, np.integer)):
                            if 0 in classes:
                                fake_idx = classes.index(0)
                            if 1 in classes:
                                real_idx = classes.index(1)

                        # строковые метки
                        if isinstance(classes[0], str):
                            for i, c in enumerate(classes):
                                if "fake" in c.lower():
                                    fake_idx = i
                                if "real" in c.lower():
                                    real_idx = i

                        # Если всё нашли — используем реальные вероятности
                        if fake_idx is not None and real_idx is not None:
                            fake_confidence = float(proba[fake_idx])
                            real_confidence = float(proba[real_idx])
                            confidence = (fake_confidence if is_fake else real_confidence) * 100
                        else:
                            # Если не разобрались с классами — оставляем 90%
                            confidence = 90.0
                    except Exception:
                        # Любая ошибка в predict_proba — не ломаем UI
                        confidence = 90.0
                else:
                    # Если predict_proba нет — просто даём 90/10
                    confidence = 90.0

                # 4. ГЕНЕРИРУЕМ ОБЪЯСНЕНИЕ (КЛЮЧЕВЫЕ СЛОВА)
                explanation = generate_explanation(input_text, is_fake, confidence)

                # 5. ОТРИСОВКА
                st.markdown("---")
                st.header("📊 Analysis Results")

                # Prediction box
                if is_fake:
                    st.markdown(f"""
                    <div class="prediction-box fake-news">
                        <div class="prediction-icon">❌</div>
                        <div class="prediction-label">FAKE NEWS</div>
                        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box real-news">
                        <div class="prediction-icon">✅</div>
                        <div class="prediction-label">REAL NEWS</div>
                        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Полоса уверенности
                st.markdown("### Confidence Level")
                st.progress(min(max(confidence / 100, 0.0), 1.0))

                # Метрики
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Prediction", "FAKE" if is_fake else "REAL")
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                with col3:
                    st.metric("Word Count", explanation['stats']['word_count'])
                with col4:
                    st.metric("Unique Words", explanation['stats']['unique_words'])

                # Детальные вкладки
                st.markdown("---")
                tab1, tab2, tab3 = st.tabs(["🎯 Key Indicators", "📊 Text Analysis", "🔍 Details"])

                with tab1:
                    st.markdown("### Why This Prediction?")

                    col_a, col_b = st.columns(2)

                    with col_a:
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

                    with col_b:
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

                with tab2:
                    st.markdown("### Text Statistics")
                    st.write(explanation['stats'])
                    st.markdown("### Sample of processed words")
                    st.write(explanation['all_words'])

                with tab3:
                    st.markdown("### Raw model output (for debugging)")
                    st.write({"raw_prediction": raw_pred, "is_fake": is_fake})

else:
    st.info(
        "Enter a news text on the left and click **Analyze Article** "
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
