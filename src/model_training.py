import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords')

# === Load dataset ===
data = pd.read_csv("data/amazon_reviews.csv", header=None, encoding='latin-1', low_memory=False)

# Assign correct column names (based on your sample)
data.columns = [
    "id", "product_id", "user_id", "profile_name",
    "helpfulness_numerator", "helpfulness_denominator",
    "score", "time", "summary", "text"
]

# === Data Cleaning ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-letter chars
    text = " ".join(
        word for word in text.split() if word not in stopwords.words('english')
    )
    return text

# Drop missing score or text
data = data.dropna(subset=['score', 'text'])

# Clean text
data["clean_text"] = data["text"].apply(clean_text)

# === Label Sentiment ===
def label_sentiment(score):
    try:
        score = int(score)
        if score >= 4:
            return "positive"
        elif score == 3:
            return "neutral"
        else:
            return "negative"
    except:
        return None

data["sentiment"] = data["score"].apply(label_sentiment)
data = data.dropna(subset=["sentiment"])  # remove invalid rows

# Remove empty clean text
data = data[data["clean_text"].str.strip() != ""]

if data.empty:
    raise ValueError("Dataset is empty after cleaning! Please check your CSV content.")

# === Train model ===
X = data["clean_text"]
y = data["sentiment"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")

# === Save model and vectorizer ===
with open("src/sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("src/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully in 'src/' folder.")

