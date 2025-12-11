# 🩺 Diabetes Prediction Benchmark

This project performs a comprehensive comparison of various Machine Learning algorithms to predict the onset of diabetes. The goal is to identify the most accurate classification model for medical risk assessment using the Pima Indians Diabetes Dataset.

## 🚀 Project Overview
We trained and evaluated multiple models to find the best performer for diabetes classification. The analysis includes:
* **Data Preprocessing:** Handling missing values and scaling.
* **Model Comparison:** Testing 8+ algorithms including XGBoost, CatBoost, and Random Forest.
* **Hyperparameter Tuning:** Optimizing models for better accuracy.

## 🏆 Model Comparison Results
The following algorithms were benchmarked:
* **Ensemble Methods:** XGBoost, CatBoost, LightGBM, Random Forest
* **Linear Models:** Logistic Regression
* **Other:** KNN, SVM, Naive Bayes, CART

**Best Performing Model:** `Random Forest` / `XGBClassifier` (Accuracy: ~75-76%)

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-learn, Statsmodels
* **Advanced Boosting:** XGBoost, CatBoost, LightGBM
* **Visualization:** Seaborn, Matplotlib

## 💻 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
