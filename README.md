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
```

---

## ⚙️ Installation & Setup

1️⃣ Clone the repository
```python
git clone https://github.com/mridulrosario25-hub/spam_detector.git
cd spam_detector
```

2️⃣ Install dependencies
```python
pip install pandas scikit-learn streamlit matplotlib seaborn
```

3️⃣ Run the app
```python
streamlit run app.py
```
or
```python
python -m streamlit run app.py
```

---

## 🧪 Models Used
### 🔹 Naive Bayes

- Fast and efficient for text classification
- Performs well on sparse data
- Produces conservative probability estimates

### 🔹 Logistic Regression

- Better calibrated probabilities
- More confident predictions for clear spam messages
- Used with `max_iter=1000` for convergence

Users can switch between models using the UI.

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

A **dark-mode confusion matrix** is displayed for better readability and UI consistency.

---

## ⚠️ Handling Uncertainty
Instead of forcing binary predictions, the app introduces an “Uncertain” state for low-confidence cases.

This reflects real-world ML behavior, where ambiguous inputs should not be overconfidently classified.
Low-confidence predictions are surfaced transparently to the user.

---

## 🖥️ User Interface
The streamlit-based interface includes:
  - Text input for message checking
  - Model selection toggle (Naive Bayes / Logisitic Regression)
  - Confidence bars using **predict_proba**
  - Expandable sections for:
      - Model accuracy
      - Classification report
      - Confusion matrix
   
---

## 📜 License
This project is intended for learning, experimentation, and portfolio purposes.

---
