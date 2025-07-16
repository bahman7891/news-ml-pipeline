# News ML Pipeline

A fully automated machine learning pipeline that fetches, processes, clusters, and visualizes the latest news articles using **NewsAPI**. The pipeline runs daily via **GitHub Actions** and updates a live dashboard hosted on **GitHub Pages**.

## Overview

This project uses a CI/CD pipeline to:
1. Fetch recent news articles from NewsAPI.
2. Clean and vectorize the text data.
3. Perform clustering (e.g., KMeans or DBSCAN) to group news articles into topics.
4. Generate a visual HTML dashboard summarizing the top clusters.
5. Deploy the dashboard automatically to GitHub Pages.

## Live Demo

[News Dashboard](https://bahman7891.github.io)

## Features

- Automated news ingestion using NewsAPI
- NLP pipeline: cleaning, tokenization, and TF-IDF vectorization
- Topic clustering using scikit-learn
- Dashboard generation in HTML with topic summaries
- CI/CD with GitHub Actions for daily updates
- Hosted on GitHub Pages

## Project Structure

