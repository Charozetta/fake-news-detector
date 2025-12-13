"""
FAKE NEWS DETECTOR — STREAMLIT DEMO APP
======================================

Simple, stable demo application for fake news detection.
Model: scikit-learn Pipeline (TF-IDF + classifier)
Classes: 0 = FAKE, 1 = REAL
"""

import streamlit as st
import numpy as np
import re
import json
import os
import glob
import joblib

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Fake News Detector (Demo)",
    page_icon="🔍",
    layout="wide",
)

# ============================================================================
# BASIC STYLING
# ============================================================================

st.markdown(
    """
<style>
.prediction-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.fake-badge {
    background: rgba(220,53,69,0.15);
    color: #dc3545;
    border: 1px solid #dc3545;
}
.real-badge {
    background: rgba(40,167,69,0.15);
    color: #28a745;
    border: 1px solid #28a745;
}
.confidence-bar {
    height: 18px;
    border-radius: 999px;
    background-color: #e9ecef;
    overflow: hidden;
}
.confidence-fill-fake {
    height: 100%;
    background: linear-gradient(90deg, #ff6b6b, #dc3545);
}
.confidence-fill-real {
    height: 100%;
    background: linear-gradient(90deg, #51cf66, #28a745);
}
.keyword {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    background: #f1f3f5;
    margin: 0.15rem;
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# MODEL LOADING
# ============================================================================


@st.cache_resource
def load_model():
    files = glob.glob("fake_news_pipeline_*.pkl")
    if not files:
        st.error("❌ No model file found (fake_news_pipeline_*.pkl).")
        st.stop()

    model_path = sorted(files)[-1]
    model = joblib.load(model_path)
    return model, model_path


with st.spinner("🔄 Loading model..."):
    pipeline, model_path = load_model()

# ============================================================================
# TEXT PREPROCESSING (KEEP SIMPLE & CONSISTENT)
# ============================================================================


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())


# ============================================================================
# EXPLANATION (KEYWORD-BASED — DEMO ONLY)
# ============================================================================


def generate_explanation(text: str):
    processed = preprocess_text(text)

    fake_keywords = [
        "breaking",
        "shocking",
        "exclusive",
        "secret",
        "conspiracy",
        "they don't want you",
        "click",
        "share",
        "rumor",
    ]

    real_keywords = [
        "according",
        "report",
        "official",
        "data",
        "study",
        "research",
        "statement",
        "confirmed",
        "source",
    ]

    found_fake = [kw for kw in fake_keywords if kw in processed]
    found_real = [kw for kw in real_keywords if kw in processed]

    return found_fake, found_real


# ============================================================================
# UI — SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Demo info")
    st.markdown(f"**Model file:** `{os.path.basename(model_path)}`")
    st.markdown("**Classes:** 0 = FAKE, 1 = REAL")
    st.markdown("---")

    example_fake = st.checkbox("Insert FAKE example")
    example_real = st.checkbox("Insert REAL example")

# ============================================================================
# UI — MAIN
# ============================================================================

st.title("🔍 Real-time Fake News Classifier (Demo)")
st.write(
    "Paste a news article or statement below. "
    "The model predicts whether it is **FAKE** or **REAL**, "
    "with confidence and simple keyword-based explanation."
)

col_left, col_right = st.columns([1.4, 1])

# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------

with col_left:
    default_text = ""

    if example_fake:
        default_text = (
            "BREAKING: Scientists discover a secret cure for all diseases, "
            "but powerful corporations are hiding the truth to protect profits."
        )
    elif example_real:
        default_text = (
            "According to an official report published by the World Health Organization, "
            "vaccination programs have significantly reduced measles cases worldwide."
        )

    text_input = st.text_area(
        "Input text",
        value=default_text,
        height=260,
        placeholder="Paste or type a news article here...",
    )

    analyze = st.button("🔍 Analyze text")

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

with col_right:
    st.markdown("### 📊 Prediction")

    if analyze:
        if not text_input or len(text_input.strip()) < 50:
            st.error("❌ Please enter at least 50 characters.")
        else:
            processed = preprocess_text(text_input)

            if len(processed.split()) < 5:
                st.error("❌ Text too short after preprocessing.")
            else:
                # Prediction
                raw_pred = pipeline.predict([processed])[0]

                # Mapping
                is_fake = int(raw_pred) == 0
                label = "FAKE news" if is_fake else "REAL news"
                emoji = "🚨" if is_fake else "✅"
                badge_class = "fake-badge" if is_fake else "real-badge"

                # Confidence
                confidence = 90.0
                if hasattr(pipeline, "predict_proba"):
                    try:
                        proba = pipeline.predict_proba([processed])[0]
                        confidence = (proba[0] if is_fake else proba[1]) * 100
                    except Exception:
                        confidence = 90.0

                # Badge
                st.markdown(
                    f"""
                    <div class="prediction-badge {badge_class}">
                        {emoji} {label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Confidence bar
                fill_class = (
                    "confidence-fill-fake" if is_fake else "confidence-fill-real"
                )
                st.markdown(
                    f"""
                    <div>Model confidence: <strong>{confidence:.1f}%</strong></div>
                    <div class="confidence-bar">
                        <div class="{fill_class}" style="width:{confidence:.1f}%"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Explanation
                fake_kw, real_kw = generate_explanation(text_input)

                st.markdown("### 🔎 Explanation (keywords — demo)")

                if not fake_kw and not real_kw:
                    st.write(
                        "No indicative keywords found. "
                        "The model relied on general text patterns."
                    )

                if fake_kw:
                    st.markdown("**Fake-related keywords:**")
                    st.markdown(
                        "".join(
                            f"<span class='keyword'>{kw}</span>" for kw in fake_kw
                        ),
                        unsafe_allow_html=True,
                    )

                if real_kw:
                    st.markdown("**Real-related keywords:**")
                    st.markdown(
                        "".join(
                            f"<span class='keyword'>{kw}</span>" for kw in real_kw
                        ),
                        unsafe_allow_html=True,
                    )

                # Debug (can be removed after demo)
                st.markdown("### 🧪 Debug")
                st.write(
                    {
                        "raw_prediction": int(raw_pred),
                        "interpreted_as_fake": is_fake,
                    }
                )
    else:
        st.info("Enter text and click **Analyze text**.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    """
---
This is a **demo application** for educational purposes.  
The explanation is heuristic and not a true model interpretation.
"""
)
