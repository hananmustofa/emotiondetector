# 🧠 EmotionDetector

EmotionDetector is an AI-powered application designed to detect and classify human emotions in text messages. By leveraging natural language processing (NLP) and machine learning techniques, it can accurately identify emotions such as **joy**, **sadness**, **anger**, **fear**, **love**, and **surprise**.

## 🚀 Project Overview

In today’s digital communication, understanding emotional tone is essential — especially for applications like:

- Chatbots and virtual assistants
- Mental health monitoring tools
- Customer service sentiment tracking
- Social media monitoring

This project uses a **TF-IDF vectorizer** to transform textual data and a **Support Vector Machine (SVM)** classifier to predict emotions. The model is trained on over 16,000 labeled examples and achieves over 87% accuracy.

## 🔧 Features

- 💬 Classifies text into six emotional categories
- 📊 Built using scikit-learn and TF-IDF for interpretability
- 🌐 Simple and responsive web interface using Streamlit
- 🧪 Trained and tested on real-world dataset from Kaggle
- 📈 Tracks emotion history over time (bonus feature)

## 📂 File Structure

```
emotiondetector/
├── main.py              # Streamlit app interface
├── model.pkl            # Trained SVM model (optional)
├── emotion_dataset.csv  # Labeled training data
├── utils.py             # Helper functions for preprocessing/prediction
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

## 🛠️ Installation & Usage

1. Clone this repository  
```bash
git clone https://github.com/hananmustofa/emotiondetector.git
cd emotiondetector
```

2. Install required packages  
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app  
```bash
streamlit run main.py
```

4. Try it out  
Enter any sentence like:  
> “I am feeling amazing today!”  
and see the predicted emotion instantly.

## 📈 Dataset

We use the open-source [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset?resource=download), which contains labeled text phrases across six emotion classes.

## 🔬 Technologies Used

- Python 3.x
- Streamlit
- scikit-learn
- Pandas, NumPy
- TF-IDF Vectorization
- SVM Classifier

## 🧠 Future Improvements

- Add support for multiple languages
- Improve subtle emotion detection (e.g., fear vs. surprise)
- Integrate with messaging apps and social media platforms
- Use deep learning (e.g., BERT) for more nuanced understanding

## 📚 References

- [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset?resource=download)
- [News-Medical: Multi-modal AI for Emotion Detection](https://www.news-medical.net/news/20240923/Multi-modal-AI-systems-for-enhanced-emotion-detection-and-understanding.aspx)
- [PMC: Emotion Recognition Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC11223560/)


## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
