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
        raise RuntimeError(" NEWS_API_KEY is not set!")

    print(f"🔑 Using API Key: {NEWS_API_KEY[:4]}***")

    # Fetch top headlines
    response = requests.get(f"{NEWS_URL}&apiKey={NEWS_API_KEY}")
    data = response.json()

    if response.status_code != 200:
        raise RuntimeError(f" API call failed: {data.get('message', response.status_code)}")

    articles = data.get("articles", [])
    if not articles:
        raise RuntimeError(" No articles returned from NewsAPI. Dashboard will not be generated.")

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

    # Print a preview in GitHub logs
    print("📰 Top 5 Articles:")
    print(df[["source", "title", "publishedAt"]].head())

    with mlflow.start_run(run_name="fetch_news"):
        mlflow.log_param("newsapi_country", "us")
        mlflow.log_param("pageSize", 100)
        mlflow.log_metric("fetched_articles", len(df))
        mlflow.log_param("fetch_status", "success")

        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        os.makedirs("data", exist_ok=True)
        csv_path = f"data/news_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        # Generate dashboard
        html_table = df[["source", "title", "publishedAt", "url"]].to_html(
            index=False, render_links=True, escape=False, border=1
        )

        html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>News Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 40px; }}
        h1 {{ font-size: 28px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; border: 1px solid #ccc; }}
        th {{ background-color: #f2f2f2; }}
        footer {{ margin-top: 40px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <h1> Latest News Dashboard</h1>
    <p>Updated: {timestamp}</p>
    {html_table}
    <footer>Auto-generated from NewsAPI on {timestamp}</footer>
</body>
</html>
"""

        html_path = "news_dashboard.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_page)

        mlflow.log_artifact(html_path)
        print(f" Dashboard saved at {os.path.abspath(html_path)}")

if __name__ == "__main__":
    fetch_news()
