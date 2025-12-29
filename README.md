# NLP_Bullying_Detector
# 🛡️ Cyberbullying Detection System

A machine learning application that identifies cyberbullying in text using Natural Language Processing. Built with Logistic Regression and TF-IDF vectorization, achieving 82.4% accuracy across multiple harassment categories.

## 🎯 Detection Categories

- **Religion-based** harassment
- **Age-based** discrimination  
- **Ethnicity-based** harassment
- **Gender-based** harassment
- **General cyberbullying**
- **Non-harmful content**

## 📊 Model Metrics

- **Accuracy**: 82.4%
- **Training Data**: 47,692 labeled tweets
- **Features**: 5,000 TF-IDF vectors
- **Algorithm**: Logistic Regression

## 🛠️ Technology Stack

- Python 3.8+
- Scikit-learn
- NLTK
- Streamlit
- Pandas & NumPy

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/adeel-iqbal/cyberbullying-analyzer.git
cd cyberbullying-analyzer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Run application
streamlit run app.py
```

## 💻 Usage Example

```python
import joblib
from preprocess import clean_tweet

# Load models
model = joblib.load("cyberbullying_lr_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Analyze text
text = "Your text here"
cleaned = clean_tweet(text)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]
category = label_encoder.inverse_transform([prediction])[0]

print(f"Detected: {category}")
```

## 📁 Project Structure

```
cyberbullying-analyzer/
├── app.py                          # Streamlit interface
├── preprocess.py                   # Text preprocessing
├── cyberbullying_analyzer.ipynb    # Training notebook
├── cyberbullying_tweets.csv        # Dataset
├── cyberbullying_lr_model.pkl      # Trained model
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
├── label_encoder.pkl               # Label encoder
└── requirements.txt                # Dependencies
```

## ⚠️ Limitations

- Trained primarily on English Twitter data
- Lower performance on general cyberbullying detection
- May not capture sarcasm or complex context

## 📈 Future Improvements

- Multi-language support
- Deep learning models (BERT, transformers)
- Real-time social media monitoring
- Severity scoring
- API deployment

## 📄 License

This project is open source and available for educational purposes.
