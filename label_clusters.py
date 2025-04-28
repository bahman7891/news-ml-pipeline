# label_clusters.py

import pandas as pd
import glob
from sklearn.feature_extraction.text import TfidfVectorizer

def label_clusters(df, text_column='text', cluster_column='cluster', top_n=5):
    cluster_labels = {}
    for cluster_id in sorted(df[cluster_column].unique()):
        cluster_texts = df[df[cluster_column] == cluster_id][text_column].dropna().tolist()

        if len(cluster_texts) == 0:
            cluster_labels[cluster_id] = ["(empty cluster)"]
            continue

        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(cluster_texts)
        
        feature_array = tfidf.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        sorted_indices = tfidf_scores.argsort()[::-1]

        top_keywords = [feature_array[i] for i in sorted_indices[:top_n]]
        cluster_labels[cluster_id] = top_keywords

    return cluster_labels

def main():
    # Find latest clustered file
    clustered_files = sorted(glob.glob('data/clustered_news_*.csv'))
    if not clustered_files:
        raise FileNotFoundError("No clustered news files found!")
    
    latest_clustered_file = clustered_files[-1]
    df = pd.read_csv(latest_clustered_file)

    # Figure out best text field
    if 'content' in df.columns:
        text_field = 'content'
    elif 'description' in df.columns:
        text_field = 'description'
    else:
        text_field = 'title'

    # Label clusters
    cluster_labels = label_clusters(df, text_column=text_field, cluster_column='cluster')

    # Save enhanced dataframe
    df['cluster_label'] = df['cluster'].map(lambda x: ', '.join(cluster_labels[x]))
    labeled_output_path = latest_clustered_file.replace('clustered_', 'labeled_clustered_')
    df.to_csv(labeled_output_path, index=False)

    print(f"\n✅ Labeled clusters and saved to {labeled_output_path}!")
    print(df[['title', 'cluster', 'cluster_label']].head())

if __name__ == "__main__":
    main()
