
import os
import requests
import pandas as pd
from datetime import datetime

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("data", exist_ok=True)
    df.to_csv(f"data/news_{timestamp}.csv", index=False)
    print(f"Saved news data to data/news_{timestamp}.csv")

if __name__ == "__main__":
    fetch_news()
