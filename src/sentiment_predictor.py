import pickle
from src.data_preprocessing import clean_text

def load_model():
    with open("models/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    prediction = model.predict(X)[0]
    return prediction
