# 📩 Spam Detection using Machine Learning

A web-based Spam Detection application built using **Machine Learning and Streamlit**.  
The app classifies text messages as **Spam**, **Not Spam**, or **Uncertain**, while also showing **model confidence** and **performance metrics**.

This project compares two popular ML models:
- **Naive Bayes**
- **Logistic Regression**

---

## 🚀 Features

- 🔍 Real-time spam detection from user input
- 🧠 Model comparison (Naive Bayes vs Logistic Regression)
- 📊 Confidence bars showing prediction certainty
- ⚠️ Explicit handling of low-confidence (uncertain) predictions
- 📈 Model evaluation:
  - Accuracy
  - Classification report
  - Confusion matrix (dark mode)
- 🎛️ Interactive UI built with Streamlit

---

## 🛠️ Tech Stack

- **Python**
- **Pandas** – data handling
- **Scikit-learn** – ML models & evaluation
- **TF-IDF Vectorizer** – text feature extraction
- **Streamlit** – web UI
- **Matplotlib & Seaborn** – visualizations

---

## 📂 Dataset

The dataset contains labeled text messages:
- `ham` → Not Spam
- `spam` → Spam

Make sure the dataset CSV file path is updated correctly in the code:
```python
pd.read_csv("path/to/spam_ham_dataset.csv")
