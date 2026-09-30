from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import sqlite3
import re
import os
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

DATABASE = "sentiment_reviews.db"
DATASET = "product_review_sentiment_500.csv"

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Your existing preprocessing code below
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.strip()

    return text


def init_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def load_dataset():
    if not os.path.exists(DATASET):
        raise FileNotFoundError(
            "product_review_sentiment_500.csv not found"
        )

    df = pd.read_csv(DATASET)

    df = df.dropna(subset=["review", "sentiment"])

    df["cleaned_review"] = (
    df["review"]
    .fillna("")
    .astype(str)
    .apply(preprocess_text)
)

    return df


df = load_dataset()

X = df["cleaned_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_tfidf
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)

logistic_precision = precision_score(
    y_test,
    logistic_prediction,
    average="weighted"
)

logistic_recall = recall_score(
    y_test,
    logistic_prediction,
    average="weighted"
)

logistic_f1 = f1_score(
    y_test,
    logistic_prediction,
    average="weighted"
)

naive_bayes_model = MultinomialNB()

naive_bayes_model.fit(
    X_train_tfidf,
    y_train
)

nb_prediction = naive_bayes_model.predict(
    X_test_tfidf
)

nb_accuracy = accuracy_score(
    y_test,
    nb_prediction
)

nb_precision = precision_score(
    y_test,
    nb_prediction,
    average="weighted"
)

nb_recall = recall_score(
    y_test,
    nb_prediction,
    average="weighted"
)

nb_f1 = f1_score(
    y_test,
    nb_prediction,
    average="weighted"
)

if logistic_accuracy >= nb_accuracy:
    selected_model = logistic_model
    selected_model_name = "Logistic Regression"
else:
    selected_model = naive_bayes_model
    selected_model_name = "Naive Bayes"


def save_prediction(review, sentiment, confidence, model):
    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO reviews
        (review, sentiment, confidence, model)
        VALUES (?, ?, ?, ?)
    """, (
        review,
        sentiment,
        confidence,
        model
    ))

    conn.commit()
    conn.close()


def get_database_reviews():
    conn = sqlite3.connect(DATABASE)

    df_db = pd.read_sql_query(
        """
        SELECT *
        FROM reviews
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return df_db


HTML = """
<!DOCTYPE html>
<html>
<head>

<title>Product Review Sentiment Analysis</title>

<meta name="viewport"
content="width=device-width, initial-scale=1">

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f9;
    color: #222;
}

.header {
    background: #172554;
    color: white;
    padding: 25px;
    text-align: center;
}

.header h1 {
    margin: 0;
    font-size: 30px;
}

.header p {
    margin-top: 8px;
    font-size: 16px;
}

.container {
    width: 92%;
    max-width: 1200px;
    margin: 30px auto;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
}

textarea {
    width: 100%;
    height: 130px;
    padding: 15px;
    font-size: 16px;
    border: 1px solid #ccc;
    border-radius: 8px;
    resize: vertical;
}

button {
    margin-top: 15px;
    padding: 12px 25px;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 7px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    background: #1d4ed8;
}

.result {
    margin-top: 20px;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

.positive {
    background: #dcfce7;
    color: #166534;
}

.negative {
    background: #fee2e2;
    color: #991b1b;
}

.neutral {
    background: #fef3c7;
    color: #92400e;
}

.stats {
    display: grid;
    grid-template-columns:
    repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
}

.stat {
    padding: 20px;
    border-radius: 10px;
    background: #eff6ff;
    text-align: center;
}

.stat h2 {
    margin: 5px;
    color: #1d4ed8;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 12px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}

th {
    background: #172554;
    color: white;
}

.model-box {
    display: grid;
    grid-template-columns:
    repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
}

.model {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 20px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #666;
}

</style>

</head>

<body>

<div class="header">

<h1>DHANVARSHINI.V</h1>

<p>
Product Review Sentiment Analysis System
</p>

</div>

<div class="container">

<div class="card">

<h2>🔍 Analyze Product Review</h2>

<form method="POST" action="/predict">

<textarea
name="review"
placeholder="Enter your product review here..."
required
></textarea>

<br>

<button type="submit">
Analyze Sentiment
</button>

</form>

{% if result %}

<div class="result {{ result_class }}">

<h2>{{ emoji }} {{ sentiment }}</h2>

<h3>
Confidence:
{{ confidence }}%
</h3>

<p>
Model Used: {{ model }}
</p>

</div>

{% endif %}

</div>


<div class="card">

<h2>📊 Dataset Information</h2>

<div class="stats">

<div class="stat">

<h2>{{ total_reviews }}</h2>

<p>Total Reviews</p>

</div>

<div class="stat">

<h2>{{ positive }}</h2>

<p>Positive Reviews</p>

</div>

<div class="stat">

<h2>{{ negative }}</h2>

<p>Negative Reviews</p>

</div>

<div class="stat">

<h2>{{ neutral }}</h2>

<p>Neutral Reviews</p>

</div>

</div>

</div>


<div class="card">

<h2>🤖 Model Performance Comparison</h2>

<div class="model-box">

<div class="model">

<h3>Logistic Regression</h3>

<p>
Accuracy:
<strong>{{ lr_accuracy }}%</strong>
</p>

<p>
Precision:
<strong>{{ lr_precision }}%</strong>
</p>

<p>
Recall:
<strong>{{ lr_recall }}%</strong>
</p>

<p>
F1 Score:
<strong>{{ lr_f1 }}%</strong>
</p>

</div>


<div class="model">

<h3>Naive Bayes</h3>

<p>
Accuracy:
<strong>{{ nb_accuracy }}%</strong>
</p>

<p>
Precision:
<strong>{{ nb_precision }}%</strong>
</p>

<p>
Recall:
<strong>{{ nb_recall }}%</strong>
</p>

<p>
F1 Score:
<strong>{{ nb_f1 }}%</strong>
</p>

</div>

</div>

<p>
Selected model:
<strong>{{ selected_model }}</strong>
</p>

</div>


<div class="card">

<h2>📈 Sentiment Analytics Dashboard</h2>

<table>

<tr>
<th>Sentiment</th>
<th>Number of Reviews</th>
<th>Percentage</th>
</tr>

<tr>
<td>Positive</td>
<td>{{ positive }}</td>
<td>{{ positive_percent }}%</td>
</tr>

<tr>
<td>Negative</td>
<td>{{ negative }}</td>
<td>{{ negative_percent }}%</td>
</tr>

<tr>
<td>Neutral</td>
<td>{{ neutral }}</td>
<td>{{ neutral_percent }}%</td>
</tr>

</table>

</div>


<div class="card">

<h2>🗄️ Stored Predictions</h2>

<table>

<tr>

<th>ID</th>
<th>Review</th>
<th>Sentiment</th>
<th>Confidence</th>
<th>Model</th>
<th>Date</th>

</tr>

{% for row in database_rows %}

<tr>

<td>{{ row[0] }}</td>

<td>{{ row[1] }}</td>

<td>{{ row[2] }}</td>

<td>{{ row[3] }}%</td>

<td>{{ row[4] }}</td>

<td>{{ row[5] }}</td>

</tr>

{% endfor %}

</table>

</div>

</div>

<div class="footer">

Product Review Sentiment Analysis |
NLP + Machine Learning + Flask + SQLite

</div>

</body>

</html>
"""


@app.route("/", methods=["GET"])
def home():

    positive = int(
        (df["sentiment"] == "Positive").sum()
    )

    negative = int(
        (df["sentiment"] == "Negative").sum()
    )

    neutral = int(
        (df["sentiment"] == "Neutral").sum()
    )

    total = len(df)

    database_df = get_database_reviews()

    rows = database_df.values.tolist()

    return render_template_string(
        HTML,
        result=False,
        total_reviews=total,
        positive=positive,
        negative=negative,
        neutral=neutral,
        positive_percent=round(
            positive / total * 100, 2
        ),
        negative_percent=round(
            negative / total * 100, 2
        ),
        neutral_percent=round(
            neutral / total * 100, 2
        ),
        lr_accuracy=round(
            logistic_accuracy * 100, 2
        ),
        lr_precision=round(
            logistic_precision * 100, 2
        ),
        lr_recall=round(
            logistic_recall * 100, 2
        ),
        lr_f1=round(
            logistic_f1 * 100, 2
        ),
        nb_accuracy=round(
            nb_accuracy * 100, 2
        ),
        nb_precision=round(
            nb_precision * 100, 2
        ),
        nb_recall=round(
            nb_recall * 100, 2
        ),
        nb_f1=round(
            nb_f1 * 100, 2
        ),
        selected_model=selected_model_name,
        database_rows=rows
    )


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form.get("review", "")

    if not review.strip():
        return home()

    cleaned_review = preprocess_text(review)

    review_vector = vectorizer.transform(
        [cleaned_review]
    )

    prediction = selected_model.predict(
        review_vector
    )[0]

    probabilities = selected_model.predict_proba(
        review_vector
    )[0]

    confidence = max(probabilities) * 100

    save_prediction(
        review,
        prediction,
        round(confidence, 2),
        selected_model_name
    )

    if prediction == "Positive":
        result_class = "positive"
        emoji = "😊"

    elif prediction == "Negative":
        result_class = "negative"
        emoji = "😞"

    else:
        result_class = "neutral"
        emoji = "😐"

    positive = int(
        (df["sentiment"] == "Positive").sum()
    )

    negative = int(
        (df["sentiment"] == "Negative").sum()
    )

    neutral = int(
        (df["sentiment"] == "Neutral").sum()
    )

    total = len(df)

    database_df = get_database_reviews()

    rows = database_df.values.tolist()

    return render_template_string(
        HTML,
        result=True,
        sentiment=prediction,
        confidence=round(confidence, 2),
        model=selected_model_name,
        result_class=result_class,
        emoji=emoji,
        total_reviews=total,
        positive=positive,
        negative=negative,
        neutral=neutral,
        positive_percent=round(
            positive / total * 100, 2
        ),
        negative_percent=round(
            negative / total * 100, 2
        ),
        neutral_percent=round(
            neutral / total * 100, 2
        ),
        lr_accuracy=round(
            logistic_accuracy * 100, 2
        ),
        lr_precision=round(
            logistic_precision * 100, 2
        ),
        lr_recall=round(
            logistic_recall * 100, 2
        ),
        lr_f1=round(
            logistic_f1 * 100, 2
        ),
        nb_accuracy=round(
            nb_accuracy * 100, 2
        ),
        nb_precision=round(
            nb_precision * 100, 2
        ),
        nb_recall=round(
            nb_recall * 100, 2
        ),
        nb_f1=round(
            nb_f1 * 100, 2
        ),
        selected_model=selected_model_name,
        database_rows=rows
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()

    if not data or "review" not in data:
        return jsonify({
            "error": "Please provide a review"
        }), 400

    review = data["review"]

    cleaned_review = preprocess_text(review)

    review_vector = vectorizer.transform(
        [cleaned_review]
    )

    prediction = selected_model.predict(
        review_vector
    )[0]

    probabilities = selected_model.predict_proba(
        review_vector
    )[0]

    confidence = max(probabilities) * 100

    save_prediction(
        review,
        prediction,
        round(confidence, 2),
        selected_model_name
    )

    return jsonify({
        "review": review,
        "sentiment": prediction,
        "confidence": round(confidence, 2),
        "model": selected_model_name
    })


@app.route("/api/analytics", methods=["GET"])
def analytics():

    database_df = get_database_reviews()

    positive_db = int(
        (database_df["sentiment"] == "Positive").sum()
    ) if len(database_df) > 0 else 0

    negative_db = int(
        (database_df["sentiment"] == "Negative").sum()
    ) if len(database_df) > 0 else 0

    neutral_db = int(
        (database_df["sentiment"] == "Neutral").sum()
    ) if len(database_df) > 0 else 0

    return jsonify({
        "total_predictions": len(database_df),
        "positive": positive_db,
        "negative": negative_db,
        "neutral": neutral_db,
        "models": {
            "Logistic Regression": {
                "accuracy": round(
                    logistic_accuracy * 100, 2
                ),
                "precision": round(
                    logistic_precision * 100, 2
                ),
                "recall": round(
                    logistic_recall * 100, 2
                ),
                "f1_score": round(
                    logistic_f1 * 100, 2
                )
            },
            "Naive Bayes": {
                "accuracy": round(
                    nb_accuracy * 100, 2
                ),
                "precision": round(
                    nb_precision * 100, 2
                ),
                "recall": round(
                    nb_recall * 100, 2
                ),
                "f1_score": round(
                    nb_f1 * 100, 2
                )
            }
        }
    })


init_database()


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )