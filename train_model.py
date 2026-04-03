import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load CSV
data = pd.read_csv("data/amazon_reviews.csv")  # Make sure path is correct

# Convert rating to sentiment
data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x>=4 else ('neutral' if x==3 else 'negative'))
data['clean_review'] = data['review'].apply(clean_text)

# Train model
X = data['clean_review']
y = data['sentiment']

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=300)
model.fit(X_vec, y)

# Save model and vectorizer
with open("models/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
