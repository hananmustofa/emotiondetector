# main.py
import pandas as pd
import re
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import zipfile
import os
from datetime import datetime

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Emotion Detection", layout="wide")

# Initialize and train the model automatically
@st.cache_resource
def load_and_train_model():
    """Load data and train model once when app starts"""
    try:
        # Load data
        train_df = pd.read_csv("training.csv")
        val_df = pd.read_csv("validation.csv")
        test_df = pd.read_csv("test.csv")
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df.columns = ['text', 'emotion']
        
        # Clean text
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        df['clean_text'] = df['text'].apply(clean_text)
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['emotion']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearSVC()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, vectorizer, accuracy
        
    except Exception as e:
        st.error(f"Error loading data or training model: {str(e)}")
        return None, None, 0

# Load model immediately when app starts
model, vectorizer, model_accuracy = load_and_train_model()

def analyze_text(text):
    """Analyze single text input"""
    if model is None:
        st.error("Model failed to load. Please check your data files.")
        return None
    
    clean_input = clean_text(text)
    vectorized_input = vectorizer.transform([clean_input])
    return model.predict(vectorized_input)[0]

def clean_text(text):
    """Clean individual text inputs"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def track_emotions(df):
    """Visualize emotion trends over time"""
    df['date'] = pd.to_datetime(df['date'])
    df['clean_text'] = df['text'].apply(clean_text)
    X = vectorizer.transform(df['clean_text'])
    df['emotion'] = model.predict(X)
    
    emotion_counts = df.groupby([df['date'].dt.to_period('M'), 'emotion']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    emotion_counts.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Emotion Trends Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.grid(True)
    return fig

def main():
    st.title("🎭 Emotion Detection App")
    
    # Show model status
    if model is not None:
        st.sidebar.success(f"Model loaded (Accuracy: {model_accuracy:.2%})")
    else:
        st.sidebar.error("Model not loaded - check data files")
    
    # App tabs
    tab1, tab2, tab3 = st.tabs(["Real-time Analysis", "Batch Processing", "Trend Analysis"])
    
    with tab1:
        st.header("Instant Emotion Detection")
        user_input = st.text_area("Type something here:", "I'm feeling excited about this project!")
        if st.button("Detect Emotion"):
            emotion = analyze_text(user_input)
            if emotion:
                st.success(f"Detected Emotion: **{emotion}**")
                st.balloons()
    
    with tab2:
        st.header("Process Multiple Texts")
        uploaded_file = st.file_uploader("Upload CSV file with 'text' column", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            if 'text' not in batch_df.columns:
                st.error("File must contain a 'text' column")
            else:
                with st.spinner("Analyzing..."):
                    batch_df['clean_text'] = batch_df['text'].apply(clean_text)
                    X_batch = vectorizer.transform(batch_df['clean_text'])
                    batch_df['predicted_emotion'] = model.predict(X_batch)
                
                st.dataframe(batch_df)
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results",
                    csv,
                    "emotion_results.csv",
                    "text/csv"
                )
    
    with tab3:
        st.header("Emotion Timeline Analysis")
        st.info("Upload data with 'text' and 'date' columns to see trends")
        trend_file = st.file_uploader("Upload time-based data", type=["csv"])
        if trend_file is not None:
            trend_df = pd.read_csv(trend_file)
            if not all(col in trend_df.columns for col in ['text', 'date']):
                st.error("Need both 'text' and 'date' columns")
            else:
                with st.spinner("Generating trends..."):
                    fig = track_emotions(trend_df)
                st.pyplot(fig)

if __name__ == "__main__":
    main()