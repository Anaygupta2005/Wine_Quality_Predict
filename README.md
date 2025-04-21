# Wine_Quality_Predict
Unsupervised clustering analysis on red wine quality data using KMeans and data normalization. Explores the relationship between chemical properties and perceived wine quality.

# Wine Quality Clustering with KMeans

This project uses unsupervised machine learning (KMeans clustering) to analyze and group red wines based on their chemical properties. The goal is to understand whether wines with similar chemical characteristics tend to receive similar quality ratings.

## What’s in this project?

- Loads a dataset of red wines with attributes like acidity, sugar, pH, alcohol, etc.
- Drops identifiers and target columns for clustering.
- Applies data normalization to ensure fair contribution from each feature.
- Performs KMeans clustering and evaluates the optimal number of clusters using an elbow plot.
- Assigns wines to clusters and compares them with actual quality ratings using a cross-tab.

## Techniques Used

- `pandas` for data handling
- `scikit-learn` for:
  - Normalization (`Normalizer`)
  - Clustering (`KMeans`)
- `matplotlib` for visualizing inertia vs. number of clusters

## Output

- Elbow plot to identify the ideal number of clusters.
- Cluster assignments for each wine.
- Cross-tab comparing true quality ratings with predicted clusters.

## Files

- `wineQualityReds.csv`: Dataset (not included here, you’ll need to obtain it separately).
- `wine_clustering.py`: Main analysis script.

## How to Run

1. Make sure you have Python 3 and the following libraries installed:
   ```bash
   pip install pandas scikit-learn matplotlib
