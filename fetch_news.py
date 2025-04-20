
import os
import requests
import pandas as pd
from datetime import datetime
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
NEWS_URL = "https://newsapi.org/v2/top-headlines?country=us&pageSize=100"

def fetch_news():
    headers = {"Authorization": NEWS_API_KEY}
    response = requests.get(NEWS_URL, headers=headers)
    data = response.json()
    
    articles = data.get("articles", [])
    records = []
    for article in articles:
        records.append({
            "source": article["source"]["name"],
            "title": article["title"],
            "description": article["description"],
            "publishedAt": article["publishedAt"]
        })

    df = pd.DataFrame(records)

    with mlflow.start_run(run_name="fetch_news"):
        mlflow.log_param("newsapi_country", "us")
        mlflow.log_param("pageSize", 100)
        mlflow.log_metric("fetched_articles", len(df))

        if df.empty:
            mlflow.log_param("fetch_status", "skipped - empty")
            print("No articles returned from NewsAPI.")
            return

        mlflow.log_param("fetch_status", "success")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs("data", exist_ok=True)
        file_path = f"data/news_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        mlflow.log_artifact(file_path)
        print(f"Saved news data to {file_path}")

if __name__ == "__main__":
    fetch_news()
