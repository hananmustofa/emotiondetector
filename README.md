# emotiondetector
# EmotionDetector 😄😢😠

This project uses Artificial Intelligence to detect emotions in text messages using a machine learning pipeline built with Python.

## 🔍 Overview

The goal of this project is to classify text into emotional categories such as:
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise

We use **TF-IDF** for text vectorization and a **Support Vector Machine (SVM)** model for emotion classification.

## 🧠 Features

- Clean, vectorized preprocessing using TF-IDF
- Trained on a labeled emotion dataset from Kaggle
- High accuracy on common emotions
- Streamlit-based web app for easy testing and UI

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/hananmustofa/emotiondetector.git
cd emotiondetector

streamlit run main.py

📊 Dataset
We use the open-source emotion dataset from Kaggle:
Emotion Dataset by Parul Pandey

📜 License
This project is open-source under the MIT License.
