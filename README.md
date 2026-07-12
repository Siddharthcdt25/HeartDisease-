# ❤️ Heart Disease Risk Prediction

**A machine learning web application that predicts heart disease risk using clinical health metrics, built with scikit-learn and deployed with Streamlit.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://cardio-risk-ai-app.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

🔗 **[Try the Live App →](https://cardio-risk-ai-app.streamlit.app)**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🩺 Overview

Cardiovascular disease is the leading cause of death globally. Early risk assessment can help individuals seek timely medical intervention. This project builds an end-to-end machine learning pipeline — from raw clinical data to a deployed, interactive web application — that predicts a patient's risk of heart disease based on 11 clinical features.

The app takes user-provided health metrics (age, blood pressure, cholesterol, ECG results, etc.) and returns an instant risk prediction, along with actionable health recommendations.

## 🎯 Problem Statement

Given a set of clinical measurements for a patient, predict whether they are at **high risk** or **low risk** of heart disease — framed as a binary classification problem.

## 📊 Dataset

- **Source:** Clinical heart disease dataset (918 patient records)
- **Features (11):** Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise-Induced Angina, Oldpeak (ST Depression), ST Slope
- **Target:** Presence (1) or absence (0) of heart disease

## 🔄 Project Workflow
┌─────────────────────┐     ┌──────────────────────────┐     ┌─────────────────┐
│  01_EDA & Feature    │ --> │  02_Preprocessing &       │ --> │  Streamlit App   │
│  Engineering         │     │  Model Training            │     │  (app.py)        │
│  (.ipynb)            │     │  (.ipynb)                  │     │                  │
└─────────────────────┘     └──────────────────────────┘     └─────────────────┘
• Data cleaning              • Train/test split              • User input form
• Handling zero-values       • Model training (KNN)           • Real-time prediction
• One-hot encoding           • Model evaluation                • Risk interpretation
• Feature scaling            • Export .pkl artifacts

**Artifacts passed between stages:** `heart_processed.csv`, `scaler.pkl`, `KNN_heart.pkl`, `columns.pkl`

## 📈 Model Performance

> *Replace the placeholders below with your actual evaluation metrics from `02_Preprocessing_and_Trainning.ipynb`.*

| Metric | Score |
|--------|-------|
| Accuracy | `XX%` |
| Precision | `XX%` |
| Recall | `XX%` |
| F1-Score | `XX%` |

**Model:** K-Nearest Neighbors (KNN) Classifier
**Preprocessing:** StandardScaler normalization + one-hot encoding of categorical features

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **ML/Data** | scikit-learn, pandas, numpy |
| **Model Persistence** | joblib |
| **Frontend** | Streamlit |
| **Notebooks** | Jupyter |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git & GitHub |

## 📁 Project Structure

heart-disease-prediction/
│
├── 01_EDAandFeature_Engg.ipynb          # Exploratory data analysis & feature engineering
├── 02_Preprocessing_and_Trainning.ipynb  # Preprocessing, model training & evaluation
├── app.py                                # Streamlit web application
├── heart.csv                             # Raw dataset
├── KNN_heart.pkl                         # Trained KNN model
├── scaler.pkl                            # Fitted StandardScaler
├── columns.pkl                           # Expected feature columns (encoding reference)
├── requirements.txt                      # Python dependencies
├── .python-version                       # Pinned Python version for deployment
└── README.md                             # Project documentation

## 🚀 Installation & Usage

### Run locally

```bash
# Clone the repository
git clone https://github.com/Siddharthcdt25/HeartDisease-.git
cd HeartDisease-

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Use the live app

No installation needed — try it directly:
👉 **[cardio-risk-ai-app.streamlit.app](https://cardio-risk-ai-app.streamlit.app)**

## 💡 Key Learnings

Building and deploying this project involved solving some real-world engineering challenges:

- **Environment reproducibility matters:** Model artifacts pickled with one `scikit-learn` version failed to load under a different version on the deployment server — solved by pinning exact package versions in `requirements.txt`.
- **Platform fit matters:** Initially attempted deployment on Vercel, which doesn't support Streamlit's persistent WebSocket architecture — migrated to Streamlit Community Cloud, purpose-built for this.
- **Python version compatibility:** A newly-released Python version caused native library (numpy/scikit-learn) instability on the server — resolved by explicitly selecting a stable Python version at deployment time.

## 🔮 Future Improvements

- [ ] Add model comparison (Logistic Regression, Random Forest, XGBoost) with performance benchmarking
- [ ] Build a REST API (FastAPI) version for programmatic access
- [ ] Add SHAP/feature importance visualizations for prediction explainability
- [ ] Add prediction history logging
- [ ] Write unit tests for the preprocessing pipeline
- [ ] Add CI/CD pipeline with GitHub Actions

## 👤 Author

**Siddharth**
🔗 [GitHub](https://github.com/Siddharthcdt25) · 🌐 [Live Project](https://cardio-risk-ai-app.streamlit.app)

---

⭐ If you found this project interesting, consider giving it a star on GitHub!

## ⚠️ Disclaimer

This tool is built for educational and demonstration purposes only. It is **not** a certified medical diagnostic tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
