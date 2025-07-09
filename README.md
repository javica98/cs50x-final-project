# CS50x: ML Final Project
#### Video Demo:
#### Description:

# 🧠 Flask ML Model Trainer

This is a Flask-based web application that allows users to **train**, **evaluate**, **save**, and **predict** with machine learning models using uploaded CSV files. It supports **Classification**, **Regression**, and **Clustering** algorithms.

---

## 🚀 Features

- ✅ Model Training:
  - **Classification**: KNN, SVM, Decision Tree, Random Forest, Logistic Regression, Naive Bayes, Gradient Boosting
  - **Regression**: Linear Regression, Random Forest Regressor, Decision Tree Regressor
  - **Clustering**: KMeans, DBSCAN
- 📊 Evaluation Metrics:
  - **Classification**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
  - **Regression**: Mean Squared Error (MSE), R² Score
  - **Clustering**: Number of Clusters (if applicable)
- 📥 CSV Upload & Preprocessing
- 💾 Model Storage with metadata (in `.pkl` or `.joblib` format)
- 📈 Visualization of model performance (e.g., Confusion Matrix)
- 🔍 Model List & Detail View
- 📤 Predict new data using trained models

---
## 📂 Sample Data

You can find example CSV files in the `sample_data/` directory:

- `classification_example.csv`: Example for training a classifier
- `regression_example.csv`: Example for training a regression model
- `clustering_example.csv`: Example for unsupervised clustering

## 🛠️ Requirements

- Python 3.7+
- Flask
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- uuid
- base64

Install dependencies:

```bash
pip install -r requirements.txt
