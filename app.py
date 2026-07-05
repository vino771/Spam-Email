from flask import Flask, render_template, request, jsonify
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data["message"]

    cleaned = clean_text(message)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })


from flask import Flask, render_template, request, jsonify
import joblib
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    message = data["message"]

    cleaned = clean_text(message)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(debug=True)