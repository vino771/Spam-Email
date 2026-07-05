import pandas as pd
import re
import string
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download stopwords (only first time)
nltk.download('stopwords')

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

print(df.head())

# -------------------------
# Convert Labels
# -------------------------

df["label"] = df["label"].map({
    "ham":0,
    "spam":1
})

# -------------------------
# Text Cleaning
# -------------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_message"] = df["message"].apply(clean_text)

# -------------------------
# TF-IDF
# -------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_message"])

y = df["label"]

# -------------------------
# Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# -------------------------
# Test
# -------------------------

predictions = model.predict(X_test)

print("\nAccuracy")

print(accuracy_score(y_test, predictions))

print("\nClassification Report")

print(classification_report(y_test, predictions))

print("\nConfusion Matrix")

print(confusion_matrix(y_test, predictions))

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, "model/model.pkl")

joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel Saved Successfully.")