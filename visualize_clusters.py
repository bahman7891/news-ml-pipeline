# visualize_clusters_plotly.py

import pandas as pd
import glob
import plotly.express as px
import umap
from sentence_transformers import SentenceTransformer

def load_labeled_articles():
    clustered_files = sorted(glob.glob('data/labeled_clustered_news_*.csv'))
    if not clustered_files:
        raise FileNotFoundError("No labeled clustered news files found!")
    latest_file = clustered_files[-1]
    df = pd.read_csv(latest_file)
    return df

def encode_texts(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings

def main():
    df = load_labeled_articles()

    if 'content' in df.columns:
        texts = df['content'].fillna(df['description']).fillna(df['title']).tolist()
    else:
        texts = df['title'].tolist()

    print(" Encoding texts...")
    embeddings = encode_texts(texts)

    print(" Applying UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)

    df['umap_x'] = umap_embeddings[:, 0]
    df['umap_y'] = umap_embeddings[:, 1]

    # after UMAP projection
    fig = px.scatter(
        df,
        x='umap_x',
        y='umap_y',
        color='cluster',
        hover_data=['title', 'cluster_label'],
        title="UMAP Clustering of News Articles",
        color_continuous_scale=px.colors.qualitative.T10
    )

    # Bigger dots and cleaner look
    fig.update_traces(
        marker=dict(
            size=12,      
            opacity=0.8,  
            line=dict(width=0.5, color='DarkSlateGrey')
        )
    )

    # Save
    fig.write_html('data/umap_plot.html')
    print(" Saved UMAP plot to 'data/umap_plot.html'!")
    

    # Save interactive plot
    fig.write_html('data/umap_plot.html')
    print(" Saved interactive UMAP plot to 'data/umap_plot.html'!")

    # Now embed it into dashboard
    print(" Embedding plot into news_dashboard.html...")
    embed_umap_plot_into_dashboard()

def embed_umap_plot_into_dashboard():
    # Load existing dashboard
    with open('news_dashboard.html', 'r', encoding='utf-8') as f:
        dashboard_html = f.read()

    # Insert <iframe> for UMAP below the table
    iframe_code = """
    <hr>
    <h2> News Topic Clusters</h2>
    <iframe src="data/umap_plot.html" width="100%" height="600px" frameborder="0"></iframe>
    """

    # Insert before final footer or simply append
    if '</body>' in dashboard_html:
        dashboard_html = dashboard_html.replace('</body>', f'{iframe_code}\n</body>')
    else:
        dashboard_html += iframe_code

    with open('news_dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html)

    print(" Embedded UMAP plot into 'news_dashboard.html'!")

if __name__ == "__main__":
    main()
