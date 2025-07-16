# cluster_articles.py

import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
from datetime import datetime

def load_articles(news_csv_path):
    if not os.path.exists(news_csv_path):
        raise FileNotFoundError(f"File {news_csv_path} not found.")
    df = pd.read_csv(news_csv_path)
    if df.empty:
        raise ValueError("Dataframe is empty.")
    texts = df['content'].fillna(df['description']).fillna(df['title']).tolist()
    texts = [str(t)[:512] for t in texts]  # Limit to 512 chars for BERT input
    return texts, df

def encode_texts(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings

def cluster_texts(embeddings, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def save_clustered_data(df, cluster_labels, output_csv_path):
    df['cluster'] = cluster_labels   # Now naming the column 'cluster'
    df.to_csv(output_csv_path, index=False)
    print(f" Clustered data saved to {output_csv_path}")

def main():
    # Find latest news CSV
    news_files = sorted(glob.glob("data/news_*.csv"))
    if not news_files:
        raise FileNotFoundError("No news CSV files found!")
    latest_news_file = news_files[-1]  # pick the most recent

    input_path = latest_news_file
    output_path = input_path.replace("news_", "clustered_news_")
    
    texts, df = load_articles(input_path)
    embeddings = encode_texts(texts)
    cluster_labels = cluster_texts(embeddings, n_clusters=8)
    save_clustered_data(df, cluster_labels, output_path)

if __name__ == "__main__":
    main()
# cluster_articles.py

import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
from datetime import datetime

def load_articles(news_csv_path):
    if not os.path.exists(news_csv_path):
        raise FileNotFoundError(f"File {news_csv_path} not found.")
    df = pd.read_csv(news_csv_path)
    if df.empty:
        raise ValueError("Dataframe is empty.")
    texts = df['content'].fillna(df['description']).fillna(df['title']).tolist()
    texts = [str(t)[:512] for t in texts]  # Limit to 512 chars for BERT input
    return texts, df

def encode_texts(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings

def cluster_texts(embeddings, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def save_clustered_data(df, cluster_labels, output_csv_path):
    df['cluster'] = cluster_labels   #  Now naming the column 'cluster'
    df.to_csv(output_csv_path, index=False)
    print(f" Clustered data saved to {output_csv_path}")

def main():
    # Find latest news CSV
    news_files = sorted(glob.glob("data/news_*.csv"))
    if not news_files:
        raise FileNotFoundError("No news CSV files found!")
    latest_news_file = news_files[-1]  # pick the most recent

    input_path = latest_news_file
    output_path = input_path.replace("news_", "clustered_news_")
    
    texts, df = load_articles(input_path)
    embeddings = encode_texts(texts)
    cluster_labels = cluster_texts(embeddings, n_clusters=8)
    save_clustered_data(df, cluster_labels, output_path)

if __name__ == "__main__":
    main()
