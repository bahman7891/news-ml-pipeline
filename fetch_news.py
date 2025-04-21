
import os
import requests
import pandas as pd
from datetime import datetime
import mlflow

mlflow.set_tracking_uri("file:./mlruns")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_URL = "https://newsapi.org/v2/top-headlines?country=us&pageSize=100"

def fetch_news():
    headers = {"x-api-key": NEWS_API_KEY}
    response = requests.get("https://newsapi.org/v2/top-headlines?country=us&pageSize=100", headers=headers)

    #headers = {"Authorization": NEWS_API_KEY}
    #response = requests.get(NEWS_URL, headers=headers)
    #url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=100&apiKey={os.getenv('NEWSAPI_KEY')}"
    #response = requests.get(url)

    print("Status Code:", response.status_code)
    print("Response:", response.json())
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
        </body>
        </html>
        """

        with open("news_dashboard.html", "w", encoding="utf-8") as f:
            f.write(html_page)

        mlflow.log_artifact("news_dashboard.html")
        print(f"Saved news dashboard to news_dashboard.html")

if __name__ == "__main__":
    fetch_news()
