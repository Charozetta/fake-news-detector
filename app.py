"""
FAKE NEWS DETECTOR — STREAMLIT DEMO APP (LinearSVC)
==================================================

Stable demo app for fake news detection.

Assumptions (your artifacts):
- Model saved as joblib: fake_news_pipeline_YYYYMMDD_HHMMSS.pkl
- The model is a scikit-learn Pipeline:
    ('vectorizer', TfidfVectorizer(...)),
    ('classifier', LinearSVC(...))   # or another linear SVM variant
- Classes: 0 = FAKE, 1 = REAL
- Optional metadata: best_model_info.json

Notes:
- We DO NOT preprocess text before feeding it to the pipeline.
  The pipeline should receive raw text (as during training).
- LinearSVC has no predict_proba -> we compute pseudo-confidence from decision_function margin.
"""

import glob
import json
import os
import re
import numpy as np
import joblib
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Fake News Detector (Demo)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# BASIC STYLING
# ============================================================================

st.markdown(
    """
<style>
/* Layout */
main .block-container { max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem; }

/* Badges */
.prediction-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.42rem 0.95rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 700;
    margin: 0.25rem 0 0.8rem 0;
}

.fake-badge {
    background: rgba(220,53,69,0.15);
    color: #dc3545;
    border: 1px solid rgba(220,53,69,0.65);
}

.real-badge {
    background: rgba(40,167,69,0.15);
    color: #28a745;
    border: 1px solid rgba(40,167,69,0.65);
}

/* Confidence bar */
.confidence-wrap { margin-top: 0.2rem; }
.confidence-bar {
    height: 16px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
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
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }

/* Keyword chips */
.keyword {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    margin: 0.15rem 0.2rem 0 0;
    font-size: 0.82rem;
}

/* Section titles */
.section-title { margin-top: 0.5rem; margin-bottom: 0.25rem; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 1.1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# ARTIFACT LOADING
# ============================================================================

METADATA_FILE = "best_model_info.json"
MODEL_GLOB = "fake_news_pipeline_*.pkl"


@st.cache_resource
def load_pipeline_and_metadata():
    # Find model file
    files = glob.glob(MODEL_GLOB)
    if not files:
        st.error(f"❌ No model file found. Expected something like: `{MODEL_GLOB}`")
        st.stop()

    # Choose most recent by filename sorting (timestamp in name)
    model_path = sorted(files)[-1]

    # Load pipeline
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model `{model_path}`: {e}")
        st.stop()

    # Load metadata (optional)
    metadata = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    else:
        metadata = {}

    return pipeline, model_path, metadata


with st.spinner("🔄 Loading model..."):
    pipeline, model_path, metadata = load_pipeline_and_metadata()

# ============================================================================
# DEMO KEYWORD EXPLANATION (HEURISTIC ONLY)
# ============================================================================

FAKE_KEYWORDS = [
    "breaking", "shocking", "exclusive", "secret", "conspiracy",
    "they don't want you", "click", "share", "rumor", "anonymous sources",
]
REAL_KEYWORDS = [
    "according", "report", "official", "data", "study",
    "research", "statement", "confirmed", "source", "ministry",
]


def normalize_for_keywords(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_explanation_keywords(text: str):
    t = normalize_for_keywords(text)
    found_fake = [kw for kw in FAKE_KEYWORDS if kw in t]
    found_real = [kw for kw in REAL_KEYWORDS if kw in t]
    return found_fake, found_real


# ============================================================================
# CONFIDENCE FOR LinearSVC (pseudo)
# ============================================================================

def svm_pseudo_confidence(model, x_text: str):
    """
    For LinearSVC inside a Pipeline: use decision_function margin.
    Convert margin to pseudo probability via sigmoid.
    Returns: (p_real_percent, margin) where p_real_percent is in [0..100].
    """
    if not hasattr(model, "decision_function"):
        return 90.0, None

    try:
        margin = model.decision_function([x_text])
        margin = float(np.ravel(margin)[0])
        p_real = 1.0 / (1.0 + np.exp(-margin))  # sigmoid
        return float(p_real * 100.0), margin
    except Exception:
        return 90.0, None


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Demo info")

    st.markdown(f"**Model file:** `{os.path.basename(model_path)}`")

    # Metadata display (if exists)
    model_name = metadata.get("model_name", "Linear SVM (Pipeline)")
    vectorizer_name = metadata.get("vectorizer_name", "TF-IDF")

    st.markdown(f"**Model:** `{model_name}`")
    st.markdown(f"**Vectorizer:** `{vectorizer_name}`")
    st.markdown("**Classes:** `0 = FAKE`, `1 = REAL`")

    metrics = metadata.get("metrics", {})
    if metrics:
        st.markdown("---")
        st.markdown("#### 📊 Metrics (from training)")
        st.write({
            "test_accuracy": metrics.get("test_accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
        })

    st.markdown("---")
    st.markdown("#### 🧪 Insert examples")
    example_fake = st.checkbox("Insert FAKE example", value=False)
    example_real = st.checkbox("Insert REAL example", value=False)

    st.markdown("---")
    show_debug = st.checkbox("Show debug output", value=True)

# ============================================================================
# MAIN UI
# ============================================================================

st.title("🔍 Real-time Fake News Classifier (Demo)")
st.write(
    "Paste a news article or statement below. The model predicts whether it is **FAKE** or **REAL**. "
    "Confidence is shown as a **pseudo-confidence** (LinearSVC margin → sigmoid), and the explanation "
    "is a simple keyword heuristic for demo only."
)

col_left, col_right = st.columns([1.4, 1])

# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------

with col_left:
    default_text = ""

    if example_fake:
        default_text = (
            "BREAKING: Celebrity reveals a secret conspiracy controlling global events. "
            "Anonymous sources claim shocking evidence will be released soon. Share this now!"
        )
    elif example_real:
        default_text = (
            "The finance ministry said the budget deficit narrowed in the third quarter, "
            "supported by higher tax revenues and lower energy subsidies. Officials added "
            "that inflation remains stable and economic growth forecasts were revised upward."
        )

    text_input = st.text_area(
        "Input text",
        value=default_text,
        height=260,
        placeholder="Paste or type a news article here...",
    )

    analyze = st.button("🔍 Analyze text", type="primary")

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

with col_right:
    st.markdown("### 📊 Prediction")

    if analyze:
        if not text_input or len(text_input.strip()) < 50:
            st.error("❌ Please enter at least 50 characters.")
        else:
            # IMPORTANT: do NOT preprocess before pipeline
            raw_pred = pipeline.predict([text_input])[0]
            pred_int = int(raw_pred)

            is_fake = pred_int == 0
            label = "FAKE news" if is_fake else "REAL news"
            emoji = "🚨" if is_fake else "✅"
            badge_class = "fake-badge" if is_fake else "real-badge"

            # Pseudo confidence from SVM margin
            p_real, margin = svm_pseudo_confidence(pipeline, text_input)
            p_fake = 100.0 - p_real
            confidence = p_fake if is_fake else p_real

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
            fill_class = "confidence-fill-fake" if is_fake else "confidence-fill-real"
            st.markdown(
                f"""
                <div class="confidence-wrap">
                    <div>Model pseudo-confidence: <strong>{confidence:.1f}%</strong>
                    <span class="small-muted">(LinearSVC margin → sigmoid)</span></div>
                    <div class="confidence-bar">
                        <div class="{fill_class}" style="width:{confidence:.1f}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Explanation (keywords demo)
            fake_kw, real_kw = generate_explanation_keywords(text_input)

            st.markdown("### 🔎 Explanation (keywords — demo)")

            if not fake_kw and not real_kw:
                st.write(
                    "No indicative keywords found. The model relied on broader text patterns."
                )

            if fake_kw:
                st.markdown("**Fake-related keywords found:**")
                st.markdown(
                    "".join(f"<span class='keyword'>{kw}</span>" for kw in fake_kw),
                    unsafe_allow_html=True,
                )

            if real_kw:
                st.markdown("**Real-related keywords found:**")
                st.markdown(
                    "".join(f"<span class='keyword'>{kw}</span>" for kw in real_kw),
                    unsafe_allow_html=True,
                )

            if show_debug:
                st.markdown("### 🧪 Debug")
                st.write(
                    {
                        "raw_prediction": pred_int,
                        "classes_expected": {0: "FAKE", 1: "REAL"},
                        "svm_margin": None if margin is None else float(margin),
                        "pseudo_real_%": float(p_real),
                        "pseudo_fake_%": float(p_fake),
                    }
                )
    else:
        st.info("Enter text and click **Analyze text**.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown(
    """
<hr>
<div class="small-muted">
This is a <strong>demo application</strong> for educational purposes.  
The keyword explanation is heuristic and not a true model interpretation.  
LinearSVC does not output calibrated probabilities by default; the confidence shown is a pseudo-confidence derived from the decision margin.
</div>
""",
    unsafe_allow_html=True,
)
