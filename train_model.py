
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import glob
import mlflow
import mlflow.sklearn

def get_latest_news_file():
    files = glob.glob("data/news_*.csv")
    return sorted(files)[-1] if files else None

def train():
    news_file = get_latest_news_file()
    if not news_file:
        print("No news file found.")
        return
    
    df = pd.read_csv(news_file)
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["label"] = [1 if "AI" in t else 0 for t in df["text"]]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])

    with mlflow.start_run():
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_metric("num_positive_labels", df["label"].sum())
        mlflow.log_metric("dataset_size", len(df))

        pipeline.fit(df["text"], df["label"])
        print("Model trained and logged with MLflow.")

        os.makedirs("models", exist_ok=True)
        import joblib
        joblib.dump(pipeline, "models/latest_model.pkl")

        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(news_file)

if __name__ == "__main__":
    train()
