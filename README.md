# HeartDisease-# 🫀 Heart Disease Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **A comprehensive machine learning project to analyze, predict, and understand the clinical risk factors associated with heart disease using a combined multi-source patient dataset of 918 records.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Workflow](#-workflow)
- [EDA Highlights](#-eda-highlights)
- [Feature Engineering](#-feature-engineering)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Cardiovascular disease is the **#1 cause of death globally**, responsible for nearly 17.9 million deaths per year. Early detection of heart disease risk using routine clinical data can be life-saving. This project leverages machine learning to:

- Perform in-depth **Exploratory Data Analysis (EDA)** across 11 clinical features
- Engineer meaningful features to boost predictive signal
- Build and evaluate classification models for heart disease prediction
- Derive actionable clinical insights from real patient data

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| **Origin** | Combined from 5 public heart disease databases (Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog) |
| **Total Records** | **918 patients** |
| **Features** | **11 clinical attributes** |
| **Target** | `HeartDisease` — Binary: `1` (Disease Present) / `0` (No Disease) |
| **Class Distribution** | 55.3% Positive (508) / 44.7% Negative (410) — mildly imbalanced |
| **Missing Values** | No explicit nulls, but **172 zero-valued Cholesterol** entries treated as missing |

### 📋 Feature Reference

| Feature | Type | Description | Unique Values / Range |
|---------|------|-------------|----------------------|
| `Age` | Numerical | Patient age in years | 28 – 77 |
| `Sex` | Categorical | Biological sex | M (725), F (193) |
| `ChestPainType` | Categorical | Type of chest pain | ASY, NAP, ATA, TA |
| `RestingBP` | Numerical | Resting blood pressure (mm Hg) | 0 – 200 *(1 zero = anomaly)* |
| `Cholesterol` | Numerical | Serum cholesterol (mg/dl) | 0 – 603 *(172 zeros = missing)* |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl | 1 = True, 0 = False |
| `RestingECG` | Categorical | Resting ECG results | Normal, ST, LVH |
| `MaxHR` | Numerical | Maximum heart rate achieved | 60 – 202 |
| `ExerciseAngina` | Categorical | Exercise-induced angina | Y / N |
| `Oldpeak` | Numerical | ST depression (exercise vs rest) | -2.6 – 6.2 |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment | Up, Flat, Down |
| `HeartDisease` | **Target** | Presence of heart disease | 1 / 0 |

---

## 📁 Project Structure

```
heart-disease-analysis/
│
├── 📂 data/
│   ├── raw/
│   │   └── heart.csv                      # Original dataset (918 × 12)
│   └── processed/
│       └── heart_engineered.csv           # After cleaning & feature engineering
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb                       # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb       # Feature transformation & creation
│   └── 03_Modeling.ipynb                  # Model training & evaluation (upcoming)
│
├── 📂 src/
│   ├── data_preprocessing.py              # Data cleaning utilities
│   ├── feature_engineering.py             # Feature engineering pipeline
│   └── utils.py                           # Helper functions
│
├── 📂 reports/
│   └── figures/                           # EDA plots and visualizations
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔄 Workflow

```
Raw Data  →  Data Cleaning  →  EDA  →  Feature Engineering  →  Modeling  →  Evaluation
   ✅              ✅           ✅            ✅                    🔄             🔄
```

> ✅ Completed &nbsp;&nbsp;&nbsp; 🔄 In Progress

---

## 📈 EDA Highlights

Key insights uncovered from the **918-patient dataset**:

- **Class Balance**: Dataset is mildly imbalanced — 55.3% positive vs 44.7% negative, manageable without heavy resampling
- **Gender Skew**: The dataset is male-dominant (79% male), which may affect model generalization across genders
- **Chest Pain Type**: `ASY` (Asymptomatic) chest pain accounts for 54% of cases and has the strongest association with heart disease
- **Cholesterol Anomaly**: 172 records (18.7%) have `Cholesterol = 0` — physiologically invalid, treated as missing during preprocessing
- **RestingBP Anomaly**: 1 record has `RestingBP = 0`, flagged and imputed
- **Age Range**: Patients span ages 28–77 (mean ≈ 53.5), with higher disease prevalence in the 50–65 bracket
- **ST_Slope**: `Flat` slope (460 patients) showed the highest correlation with positive disease outcomes
- **MaxHR**: Patients with heart disease consistently achieved lower maximum heart rates
- **Exercise Angina**: 40.4% of patients exhibited exercise-induced angina (`Y`), a strong risk indicator
- **Oldpeak**: Higher ST depression values positively correlated with heart disease presence

> 📓 Full analysis available in [`notebooks/01_EDA.ipynb`](notebooks/01_EDA.ipynb)

---

## ⚙️ Feature Engineering

The following transformations were applied to improve model signal:

- **Missing Value Imputation**: Cholesterol zeros (172 records) replaced with median; single RestingBP zero treated similarly
- **Label Encoding**: Binary categoricals (`Sex`, `ExerciseAngina`) encoded as 0/1
- **One-Hot Encoding**: Multi-class categoricals — `ChestPainType` (4 classes), `RestingECG` (3 classes), `ST_Slope` (3 classes) — encoded with `drop_first=True`
- **Age Binning**: Age grouped into clinical brackets — `Young (<40)`, `Middle-Aged (40–55)`, `Senior (55–65)`, `Elderly (>65)`
- **Interaction Features**:
  - `Age × MaxHR` — age-adjusted cardiovascular effort index
  - `Cholesterol × RestingBP` — combined metabolic-pressure risk score
  - `Oldpeak × ST_Slope` — combined ST-segment risk signal
- **Skewness Treatment**: Log transformation applied to `Oldpeak` to reduce right skew
- **Feature Scaling**: StandardScaler applied to continuous features (`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`) for distance-based models

> 📓 Full pipeline in [`notebooks/02_Feature_Engineering.ipynb`](notebooks/02_Feature_Engineering.ipynb)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.9+` | Core programming language |
| `Pandas` | Data manipulation & analysis |
| `NumPy` | Numerical computations |
| `Matplotlib` & `Seaborn` | Data visualization |
| `Scikit-learn` | ML modeling, preprocessing & evaluation |
| `Jupyter Notebook` | Interactive development environment |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-analysis.git
cd heart-disease-analysis

# Create a virtual environment
python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

```bash
# Place your dataset in the raw data folder
cp heart.csv data/raw/heart.csv
```

### Run Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Run notebooks in order:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_Feature_Engineering.ipynb
# 3. notebooks/03_Modeling.ipynb  (upcoming)
```

---

## 📉 Results

> 🔄 *Model training is currently in progress. Results will be updated upon completion.*

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| SVM | — | — | — | — | — |
| K-Nearest Neighbors | — | — | — | — | — |

---

## 🔮 Future Work

- [ ] Complete model training, tuning & cross-validation
- [ ] Address class imbalance with SMOTE if needed post-modeling
- [ ] Add SHAP values for model explainability
- [ ] Investigate performance gap across gender subgroups (dataset is 79% male)
- [ ] Build an interactive prediction dashboard using Streamlit
- [ ] Deploy model as a REST API (FastAPI / Flask)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request


<p align="center">
  Made with ❤️ and lots of ☕ | If you found this useful, give it a ⭐!
</p>
