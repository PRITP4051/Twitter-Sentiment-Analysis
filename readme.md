# 🤖 Twitter Sentiment Analysis using Deep Learning & BERT

🚀 An end-to-end Sentiment Analysis system built using Deep Learning and Transformer-based models (BERT), trained on real-world social media data.

---

## 📌 Project Overview

This project analyzes user sentiments from social media comments and classifies them into:

- 😊 Positive
- 😡 Negative
- 😐 Neutral

Due to restrictions in Twitter API access, the dataset was collected using the **YouTube Data API**, which provides real-time user-generated comments on trending topics.

Multiple models were implemented and compared:

- RNN
- LSTM
- GRU
- **BERT (Final Model)**

👉 BERT achieved the best performance and is used for final predictions.

---

## 🌐 Live Demo

🔗 **Streamlit App:**  
👉 https://your-streamlit-link-here

---

## 📊 Dataset Details

- 📦 Size: ~8000 comments
- 🧠 Source: YouTube Data API
- 🏷️ Labels:
  - Positive → 1
  - Negative → 0
  - Neutral → 2

### 🧹 Data Cleaning & Filtering

- Removed short comments (< 3–4 words)
- Removed emoji-only / symbol-only comments
- Removed spam/repetitive text
- Removed null/empty values
- Filtered irrelevant comments

### 🏷️ Labeling Method

- Used **TextBlob polarity** for automatic sentiment labeling

---

## ⚙️ Tech Stack

- Python
- TensorFlow / PyTorch
- HuggingFace Transformers
- Scikit-learn
- Streamlit
- YouTube Data API

---

## 🧠 Models Used

### 🔹 RNN

- Simple sequential model
- ❌ Poor performance (~68%)

### 🔹 LSTM

- Handles long dependencies
- ⚠️ Overfitting observed
- ✅ ~83% accuracy

### 🔹 GRU

- Faster than LSTM
- ⚠️ Slight overfitting
- ✅ ~82–84% accuracy

### 🔹 BERT (Final Model)

- Transformer-based model
- Uses attention mechanism
- ✅ Best performance

---

## 📈 Model Performance

| Model    | Train Accuracy | Test Accuracy | F1 Score   |
| -------- | -------------- | ------------- | ---------- |
| RNN      | 87.86%         | 68.02%        | 67.81%     |
| LSTM     | 95.07%         | 83.19%        | 82.98%     |
| GRU      | 94.19%         | 82.80%        | 82.82%     |
| **BERT** | **96.29%**     | **86.28%**    | **86.33%** |

👉 BERT outperforms all models due to better contextual understanding.

---

## 🧠 Why BERT?

- Understands full sentence context
- Uses bidirectional attention
- Pre-trained on massive datasets
- Avoids vanishing gradient problem
- Better generalization than RNN-based models

---

## ⚠️ Challenges

- Noisy labels due to auto-labeling (TextBlob)
- Limited dataset (~8000 samples)
- CPU-based training (slow BERT training)
- Slight overfitting in deep models

---

## 🌐 Streamlit App Features

### 🔍 1. Sentiment Prediction

- Real-time prediction using BERT
- Confidence score display

### 📊 2. Dataset Viewer

- Explore collected dataset

### 📈 3. Model Comparison

- Accuracy, Precision, Recall, F1 visualization

---

## 📂 Project Structure
