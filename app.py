import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import json

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
# LOAD ARTIFACTS (SVM PIPELINE + OPTIONAL METADATA)
# ============================================================================

MODEL_GLOB = "fake_news_pipeline_*.pkl"
METADATA_PATH = "best_model_info.json"


@st.cache_resource
def load_pipeline_and_metadata():
    files = glob.glob(MODEL_GLOB)
    if not files:
        st.error(f"❌ Model file not found. Expected: {MODEL_GLOB}")
        st.stop()

    model_path = sorted(files)[-1]

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    metadata = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    return pipeline, model_path, metadata


pipeline, model_path, metadata = load_pipeline_and_metadata()

# ============================================================================
# LINEAR SVM PSEUDO-PROBABILITY (decision_function -> sigmoid)
# ============================================================================

def svm_pseudo_probs(pipeline_obj, text: str):
    """
    Returns:
      pred (int): 0=FAKE, 1=REAL
      fake_prob (float): 0..100 (pseudo)
      real_prob (float): 0..100 (pseudo)
      margin (float|None): decision margin
    """
    pred = int(pipeline_obj.predict([text])[0])

    margin = None
    if hasattr(pipeline_obj, "decision_function"):
        try:
            margin_raw = pipeline_obj.decision_function([text])
            margin = float(np.ravel(margin_raw)[0])
            # sigmoid -> pseudo prob REAL
            p_real = 1.0 / (1.0 + np.exp(-margin))
            p_fake = 1.0 - p_real
            real_prob = float(p_real * 100.0)
            fake_prob = float(p_fake * 100.0)
            return pred, fake_prob, real_prob, margin
        except Exception:
            pass

    # Fallback if decision_function missing for some reason
    # (keeps UI stable)
    if pred == 1:
        return pred, 10.0, 90.0, margin
    return pred, 90.0, 10.0, margin


def predict_fake_news(text: str):
    """
    Predict if a news article is fake or real. FAKE=0, REAL=1.
    Uses pipeline (TF-IDF + LinearSVC).
    """
    pred, fake_prob, real_prob, margin = svm_pseudo_probs(pipeline, text)

    label = "REAL" if pred == 1 else "FAKE"
    confidence = real_prob if pred == 1 else fake_prob

    return {
        "text": text[:200] + "..." if len(text) > 200 else text,
        "label": label,
        "confidence": confidence,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        # debug fields (optional)
        "pred_raw": pred,
        "svm_margin": margin
    }

# ============================================================================
# MAIN UI (same structure as your Logistic Regression app)
# ============================================================================

st.markdown("# 🔍 Fake News Detection System")
st.markdown("**Detect whether a news article is FAKE or REAL using Machine Learning**")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 About Model", "📚 Examples", "ℹ️ How it Works"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================

with tab1:
    st.subheader("Enter News Article Text")

    user_text = st.text_area(
        "Paste your news article here:",
        height=200,
        placeholder="Enter the news article text you want to check..."
    )

    col1, col2 = st.columns(2)

    with col1:
        predict_button = st.button("🔍 Analyze", key="predict_main")

    with col2:
        clear_button = st.button("🗑️ Clear", key="clear_main")

    if clear_button:
        st.rerun()

    if predict_button:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze!")
        elif len(user_text.strip()) < 30:
            st.warning("⚠️ Text is too short for a stable prediction. Please paste a longer snippet.")
        else:
            with st.spinner("Analyzing text..."):
                result = predict_fake_news(user_text)

            st.markdown("---")
            st.subheader("📋 Analysis Results")

            if result['label'] == 'REAL':
                st.success(f"✅ **REAL NEWS** - Confidence: {result['confidence']:.2f}%")
            else:
                st.error(f"❌ **FAKE NEWS** - Confidence: {result['confidence']:.2f}%")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("FAKE Probability", f"{result['fake_prob']:.2f}%")
            with col2:
                st.metric("REAL Probability", f"{result['real_prob']:.2f}%")

            st.subheader("📈 Probability Distribution")
            chart_data = pd.DataFrame({
                'Label': ['FAKE', 'REAL'],
                'Probability': [result['fake_prob'], result['real_prob']]
            })
            st.bar_chart(chart_data.set_index('Label'))

            # Optional Debug expander (you can remove for final demo)
            with st.expander("🧪 Debug (optional)"):
                st.write({
                    "pred_raw": result["pred_raw"],
                    "svm_margin": result["svm_margin"],
                    "model_file": os.path.basename(model_path)
                })

# ============================================================================
# TAB 2: ABOUT MODEL
# ============================================================================

with tab2:
    st.subheader("📊 Model Information")

    col1, col2 = st.columns(2)

    model_name = metadata.get("model_name", "Linear SVM (LinearSVC)")
    vectorizer_name = metadata.get("vectorizer_name", "TF-IDF")
    metrics = metadata.get("metrics", {})

    with col1:
        st.info(f"""
        **Model Architecture:**
        - Algorithm: {model_name}
        - Feature Extraction: {vectorizer_name}
        - Training Data: ISOT Fake News Dataset
        - Classes: 0 = FAKE, 1 = REAL
        - Artifact: {os.path.basename(model_path)}
        """)

    # If you want hardcoded “pretty” metrics like LR demo, you can keep them.
    # But we will display what is in best_model_info.json if present.
    with col2:
        if metrics:
            st.success(f"""
            **Model Performance (from training):**
            - Accuracy: {metrics.get("test_accuracy", 0.0) * 100:.2f}%
            - Precision: {metrics.get("precision", 0.0) * 100:.2f}%
            - Recall: {metrics.get("recall", 0.0) * 100:.2f}%
            - F1-Score: {metrics.get("f1", 0.0) * 100:.2f}%
            """)
        else:
            st.success("""
            **Model Performance:**
            - Metrics not found in best_model_info.json
            (this is fine for demo — you can add them later)
            """)

    st.subheader("📈 Dataset Statistics")
    dataset_stats = {
        'Category': ['Real News', 'Fake News', 'Total'],
        'Count': [21417, 23481, 44898],
        'Percentage': ['48.2%', '51.8%', '100%']
    }
    df_stats = pd.DataFrame(dataset_stats)
    st.table(df_stats)

# ============================================================================
# TAB 3: EXAMPLES
# ============================================================================

with tab3:
    st.subheader("📚 Example Articles")

    # Real News Example (Reuters-style works best with ISOT)
    st.markdown("### ✅ Real News Example")
    real_example = (
        "The finance ministry said the budget deficit narrowed in the third quarter, "
        "supported by higher tax revenues and lower energy subsidies. Officials added "
        "that inflation remained stable and growth forecasts were revised upward."
    )
    st.write(real_example)

    if st.button("Analyze Real News Example", key="real_example_btn"):
        result = predict_fake_news(real_example)
        if result['label'] == 'REAL':
            st.success(f"✅ Prediction: **{result['label']}** (Confidence: {result['confidence']:.2f}%)")
        else:
            st.warning(f"⚠️ Prediction: **{result['label']}** (Confidence: {result['confidence']:.2f}%)")

    st.markdown("---")

    # Fake News Example
    st.markdown("### ❌ Fake News Example")
    fake_example = (
        "Breaking news alert: Celebrity announces secret government conspiracy. "
        "The star has revealed shocking evidence about a hidden organization controlling world events. "
        "Government officials deny all allegations as sources remain anonymous and unverifiable."
    )
    st.write(fake_example)

    if st.button("Analyze Fake News Example", key="fake_example_btn"):
        result = predict_fake_news(fake_example)
        if result['label'] == 'FAKE':
            st.error(f"❌ Prediction: **{result['label']}** (Confidence: {result['confidence']:.2f}%)")
        else:
            st.warning(f"⚠️ Prediction: **{result['label']}** (Confidence: {result['confidence']:.2f}%)")

# ============================================================================
# TAB 4: HOW IT WORKS
# ============================================================================

with tab4:
    st.subheader("❓ How Does It Work?")
    st.markdown("""
    ### 1. **Feature Extraction**
    - TF-IDF vectorization (learned from training data)
    - Each feature represents a word/term importance in the text
    
    ### 2. **Prediction**
    - Linear SVM (LinearSVC) separates FAKE vs REAL using a linear decision boundary
    - Output is a class label: FAKE (0) or REAL (1)
    
    ### 3. **Confidence (Demo)**
    - LinearSVC does not produce calibrated probabilities by default
    - For demo UI we compute **pseudo-probabilities** from the SVM decision margin
      using a sigmoid transformation (margin → value between 0 and 1)
    
    ### 4. **Important**
    - This model detects statistical patterns in text, not truth
    - Always cross-check important information using reliable sources
    """)

    st.warning("""
    ⚠️ **Important Notes:**
    - This tool provides predictions, not absolute truths
    - Dataset bias matters (ISOT “REAL” often resembles Reuters-style reporting)
    - Use this as a demo/educational system, not a full fact-checker
    """)

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.8rem;">
    <p>Fake News Detection System | Built with Streamlit & Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)
