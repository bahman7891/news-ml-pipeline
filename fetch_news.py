import os
import requests
import pandas as pd
from datetime import datetime
import mlflow

mlflow.set_tracking_uri("file:./mlruns")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = "https://newsapi.org/v2/top-headlines?country=us&pageSize=100"

def fetch_news():
    if not NEWS_API_KEY:
        raise RuntimeError("🚨 NEWS_API_KEY is not set!")

    print(f"Using API Key: {NEWS_API_KEY[:4]}***")  # Only print first 4 characters for safety

    # Pass key in URL directly (simpler and matches NewsAPI recommendation)
    response = requests.get(f"{NEWS_URL}&apiKey={NEWS_API_KEY}")
    data = response.json()

    if response.status_code != 200:
        raise RuntimeError(f"API call failed: {data.get('message', response.status_code)}")

    articles = data.get("articles", [])
    records = []

    for article in articles:
        records.append({
            "source": article.get("source", {}).get("name", ""),
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "content": article.get("content", ""),
            "publishedAt": article.get("publishedAt", ""),
            "url": article.get("url", "")
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

        # HTML Dashboard generation
        html_table = df[["source", "title", "publishedAt"]].to_html(
            index=False, render_links=True, escape=False, border=1
        )
        html_page = f"""
        <html>
        <head>
            <title>News Dashboard</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px 12px; border: 1px solid #ccc; }}
                th {{ background-color: #f4f4f4; }}
            </style>
        </head>
        <body>
            <h1>📰 Latest News Dashboard</h1>
            <p>Updated on: {timestamp}</p>
            {html_table}
            <footer style='margin-top:40px;font-size:12px;'>Last updated: {timestamp}</footer>
        </body>
        </html>
        """

        with open("news_dashboard.html", "w", encoding="utf-8") as f:
            f.write(html_page)

        mlflow.log_artifact("news_dashboard.html")
        print(f"✅ Saved news dashboard to news_dashboard.html")

if __name__ == "__main__":
    fetch_news()
