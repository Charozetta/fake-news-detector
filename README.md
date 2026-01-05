# Fake News Detection — Linear SVM Demo

This repository contains a **demo web application** for fake news detection using
a **Linear Support Vector Machine (Linear SVM)** model.

The application classifies news text as **FAKE** or **REAL** and is built with
**Streamlit** and **scikit-learn**.

---

## Project Overview

- **Task:** Binary text classification (Fake News Detection)
- **Classes:**
  - `0` — FAKE
  - `1` — REAL
- **Model:** Linear SVM (LinearSVC)
- **Text Representation:** TF-IDF
- **Dataset:** ISOT Fake News Dataset
- **Purpose:** Educational / demonstration
- **Original Dataset:** Can be download here: https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset

---

## Repository Structure
```
├── app.py
├── README.md
├── requirements.txt
├── ISOT_Fake_News_Detection_Project.ipynb
├── best_model_info.json
├── model_comparison_results.csv
├── deep_learning_comparison_results.csv
├── deep_learning_metadata.json
└── dl_tokenizer.pkl
```


---

## How to Run the App

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run Streamlit
```
streamlit run app.py
```
