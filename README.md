# Fake News Detector

AI-powered web application for detecting fake news using Machine Learning.

## About

This project is a machine learning-based web application that detects fake news articles with **99.52% accuracy**. Built as part of a data science coursework at HSE University (Moscow). It demonstrates the practical application of NLP and classical ML algorithms for text classification.

Original Dataset (ISOT) can be found here: https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset

** Live Demo:** [Coming Soon - Streamlit Cloud]

## Features

- **Real-time Detection** - Instant analysis of news articles
- **High Accuracy** - 99.52% accuracy using Linear SVM
- **Explainable AI** - Shows why an article is classified as fake/real
- **Keyword Analysis** - Highlights suspicious patterns and credible indicators
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Fast Performance** - Results in under 1 second

## Screenshots

*[Add your screenshots here after deployment]*

## Model Performance

### Classical Machine Learning

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Linear SVM** | **99.52%** | **99.48%** | **99.64%** | **0.9956** |
| Decision Tree | 99.47% | 99.43% | 99.59% | 0.9951 |
| Random Forest | 99.41% | 99.19% | 99.71% | 0.9945 |
| Logistic Regression | 99.25% | 99.17% | 99.45% | 0.9931 |
| Naive Bayes | 95.67% | 95.96% | 96.01% | 0.9599 |

### Deep Learning Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CNN | 99.77% | 99.79% | 99.79% | 0.9978 | 0.9996 |
| BiLSTM | 99.74% | 99.86% | 99.67% | 0.9976 | 0.9997 |
| LSTM+CNN Hybrid | 99.21% | 98.77% | 99.79% | 0.9928 | 0.9979 |
| LSTM | 97.75% | 97.74% | 98.09% | 0.9791 | 0.9966 |

### Final Comparison (Classical vs Deep Learning)

| Approach | Best Model | Accuracy | Inference Time | Model Size |
|----------|-----------|----------|----------------|------------|
| **Classical ML** | **Linear SVM** | **99.52%** | **<1ms** | **~5MB** |
| Deep Learning | CNN | 99.77% | ~20ms | ~150MB |

**Decision:** Linear SVM selected for production due to optimal balance between accuracy (99.52%), speed (<1ms), and resource efficiency.

**Conclusion:** Linear SVM was chosen for production deployment due to optimal balance between accuracy and computational efficiency.

## Technologies Used

- **Python 3.9+**
- **Machine Learning:** scikit-learn
- **Web Framework:** Streamlit
- **Data Processing:** pandas, numpy
- **NLP:** TF-IDF vectorization

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Charozetta/fake-news-detector.git
cd fake-news-detector
```

2. **Install dependencies:**
```bash
pip install -r requirements_app.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser:**
```
http://localhost:8501
```

## Project Structure

```
fake-news-detector/
├── app.py                              # Main Streamlit application
├── requirements_app.txt                # Python dependencies
├── fake_news_pipeline_*.pkl            # Trained ML model (Linear SVM)
├── best_model_info.json                # Model metadata
├── deeplearning_metadata.json          # Deep learning results
├── model_comparison_results.csv        # Classical ML comparison
└── README.md                           # This file
```

## Usage

### Using Example Articles

1. Click the **"Examples"** button
2. Select an example (2 real news, 2 fake news)
3. Click **"Use This Example"**
4. Click **"Analyze Article"**

### Analyzing Your Own Text

1. Paste or type a news article in the text area
2. Click **"Analyze Article"**
3. View the prediction, confidence score, and explanation

### Understanding Results

**Prediction Box:**
- Green = REAL news (credible)
- Red = FAKE news (suspicious)
- Confidence score (0-100%)

**Key Indicators:**
- Shows words that indicate fake news (e.g., "BREAKING", "SHOCKING")
- Shows words that indicate real news (e.g., "Reuters", "according to")

## Dataset

**ISOT Fake News Dataset**
- **Total articles:** 44,898
- **Real news:** 21,417 (47.7%) - Reuters
- **Fake news:** 23,481 (52.3%) - various sources  
- **Time period:** 2016-2017
- **After preprocessing:** ~35,000 articles (after removing duplicates)

**Data Split:**
- Training set: 80% (~28,000 articles)
- Test set: 20% (~7,000 articles)

**Preprocessing steps:**
- Removed duplicates (~10.9%)
- Combined title + text
- Text cleaning (URLs, HTML, special characters)
- Lowercasing and normalization
- TF-IDF vectorization (10,000 features, bigrams)

## Methodology

The project follows a comprehensive 7-part workflow:

### Part 1: Exploratory Data Analysis (EDA)
- Dataset loading and combination
- Statistical analysis (44,898 articles)
- Text length distributions
- Class balance examination
- Duplicate detection

### Part 2: Text Preprocessing
- Title + text combination
- Comprehensive cleaning pipeline:
  - Lowercasing
  - URL and HTML removal
  - Special character removal
  - Whitespace normalization
- Preprocessing statistics and validation

### Part 3: Feature Engineering & Vectorization
- Train-test split (80/20)
- **TF-IDF Vectorization** (selected):
  - Max features: 10,000
  - N-grams: (1, 2)
  - Vocabulary: 10,000 words
- Alternative methods tested:
  - Bag of Words
  - Word2Vec

### Part 4: Classical Machine Learning
- 5 algorithms evaluated:
  - Linear SVM ← **Selected**
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Naive Bayes
- Comprehensive metrics evaluation
- Model comparison and selection
- Pipeline building and artifact saving

### Part 5: Unsupervised Learning
- **Clustering:**
  - K-means optimization
  - Silhouette analysis
  - Achieved ~60-65% accuracy
- **Topic Modeling:**
  - LDA (5 topics identified)
  - LSA analysis
- **Dimensionality Reduction:**
  - PCA (<3% variance explained)
  - t-SNE visualization

### Part 6: (Skipped - directly to Part 7)

### Part 7: Deep Learning Models
- 4 architectures implemented:
  - **CNN** ← Highest accuracy (99.77%)
  - **BiLSTM** ← Best precision (99.86%)
  - LSTM+CNN Hybrid
  - LSTM
- Configuration:
  - Vocabulary: 10,000 words
  - Sequence length: 200 tokens
  - Embedding dimension: 128
  - Batch size: 64
  - Epochs: 5 (with early stopping)
- Model comparison and best model selection
- Tokenizer and metadata saving

### Final Selection Criteria
**Linear SVM** chosen for production based on:
- Highest accuracy among classical ML (99.52%)
- Ultra-fast inference (<1ms vs CNN's 20ms)
- Small model size (5MB vs CNN's 150MB)
- Easy deployment and maintenance
- Sufficient accuracy for the task

## Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy from repository
4. Get live URL

**Live Demo:** [Coming Soon]

## Results & Findings

### Key Insights

1. **Classical ML vs Deep Learning:**
   - **Linear SVM:** 99.52% accuracy, <1ms inference, 5MB model
   - **Best DL (CNN):** 99.77% accuracy, 20ms inference, 150MB model
   - **Trade-off:** +0.25% accuracy vs 20x slower and 30x larger model

2. **Best Classical ML Models:**
   - Top 3 models all achieved >99.4% accuracy
   - Linear SVM (99.52%) slightly outperformed Decision Tree (99.47%)
   - Random Forest (99.41%) showed excellent recall (99.71%)

3. **Deep Learning Performance:**
   - CNN achieved highest overall accuracy (99.77%)
   - BiLSTM showed best precision (99.86%) 
   - LSTM+CNN Hybrid underperformed (99.21%)
   - Simple LSTM struggled most (97.75%)

4. **Feature Importance:**
   - Sensational keywords ("BREAKING", "SHOCKING", "REVEALED") strongly indicate fake news
   - Source attribution ("Reuters", "according to", "officials said") indicates real news
   - Writing style and vocabulary differ significantly between classes

5. **Production Considerations:**
   - **Linear SVM optimal** for real-time applications (speed + accuracy)
   - **CNN better** for maximum accuracy when latency not critical
   - **Decision Tree best** for interpretability and explainability

6. **Unsupervised Learning:**
   - Clustering achieved ~60-65% accuracy (insufficient alone)
   - Topic modeling revealed 5 main themes
   - High dimensionality challenges (PCA explained <3% variance)

## Limitations

1. **Training Data Bias:**
   - Trained on 2016-2017 data
   - May not detect newer fake news patterns
   - English language only

2. **Context Limitations:**
   - Text analysis only (no images/videos)
   - No source verification
   - No fact-checking against databases

3. **Accuracy:**
   - 99.52% accuracy ≠ 100%
   - Always verify from multiple sources
   - 
## Academic Context

This project was developed as part of the **Unstructured Data Analysis** modul at **HSEs**. It demonstrates:

- End-to-end ML project workflow
- Text preprocessing and feature engineering
- Model comparison and selection
- Production deployment
- Explainable AI principles

## References

- **Dataset:** Ahmed, H., Traore, I., & Saad, S. (2017). "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques." ISOT Fake News Dataset.
- **Streamlit Documentation:** https://docs.streamlit.io/
- **scikit-learn Documentation:** https://scikit-learn.org/

---

** If you found this project useful, please consider giving it a star!**

*Last updated: December 2025*
