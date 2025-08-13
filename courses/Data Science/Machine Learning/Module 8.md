# Module 8: Clustering Algorithms

## What is Clustering?

Clustering is an **unsupervised learning technique** used to group similar data points into clusters. Unlike classification, clustering does not rely on predefined labels. Instead, the goal is to discover inherent patterns or structures in the data. Clustering algorithms are widely used in applications like customer segmentation, image compression, anomaly detection, and more.

### Types of Clustering Algorithms

1. **k-Means Clustering**:
    - Partitions the data into \( k \) clusters, where each data point belongs to the cluster with the nearest centroid.
    - It uses an iterative approach to minimize the variance within clusters.
2. **Hierarchical Clustering**:
    - Builds a hierarchy of clusters using either an **agglomerative** (bottom-up) or **divisive** (top-down) approach.
    - Creates a dendrogram to visualize the clustering process.

---

## Section 8.1: k-Means Clustering

### How k-Means Works:

1. **Initialization**: Select \( k \) random points as initial centroids.
2. **Assignment**: Assign each data point to the nearest centroid.
3. **Update**: Recompute the centroids based on the mean of all points in each cluster.
4. **Repeat**: Perform assignment and update steps until centroids stabilize or a maximum number of iterations is reached.

---

### Code Example 1: Basic k-Means Clustering on 2D Data

**Importing Libraries and Dataset**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

```

**Code Explanation**

In this example:

- **`make_blobs`**: Generates a dataset with 4 clusters for demonstration.
- **`X`**: Represents the feature matrix for clustering.

---

**Performing k-Means Clustering**

```python
# Apply k-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

```

**Explanation**

- **`n_clusters=4`**: Specifies the number of clusters.
- **`fit_predict`**: Fits the model and assigns cluster labels to each data point.

---

**Visualizing Clusters**

```python
# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("k-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

```

**Output Explanation**

- The scatter plot shows the clustered data points in different colors.
- Red markers represent the cluster centroids computed by k-Means.

---

### Code Example 2: Choosing the Optimal Number of Clusters (Elbow Method)

**Importing Libraries**

```python
# Elbow method to find optimal k
from sklearn.metrics import silhouette_score

```

**Elbow Method Implementation**

```python
inertia = []
silhouette_scores = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot inertia (elbow method)
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia, marker='o', label='Inertia')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.legend()
plt.show()

# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, marker='o', label='Silhouette Score', color='green')
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Score")
plt.legend()
plt.show()

```

**Explanation**

- **Inertia**: Measures the compactness of clusters. Lower inertia suggests better-defined clusters.
- **Silhouette Score**: Evaluates the quality of clustering. A higher score indicates better clustering.

---

## Section 8.2: Hierarchical Clustering

### How Hierarchical Clustering Works:

1. **Agglomerative Clustering**:
    - Each data point starts as its own cluster.
    - Pairs of clusters are merged based on similarity until a single cluster remains.
2. **Divisive Clustering**:
    - Starts with a single cluster containing all data points.
    - Splits the cluster iteratively into smaller clusters.

---

### Code Example 1: Agglomerative Clustering

**Importing Libraries and Dataset**

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=42)

```

**Performing Agglomerative Clustering**

```python
# Apply agglomerative clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_agg = agg_clustering.fit_predict(X)

```

**Explanation**

- **`affinity='euclidean'`**: Specifies the distance metric.
- **`linkage='ward'`**: Minimizes variance between merged clusters.

---

**Visualizing Clusters and Dendrogram**

```python
# Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='rainbow', s=50)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Dendrogram
linked = linkage(X, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

```

**Output Explanation**

- The scatter plot shows the clusters identified by hierarchical clustering.
- The dendrogram visualizes the merging process at each step.

---

### Code Example 2: Comparing k-Means and Hierarchical Clustering

**Comparison**

```python
# k-Means
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='coolwarm', s=50)
plt.title("k-Means Clustering")
plt.show()

# Hierarchical Clustering
plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='coolwarm', s=50)
plt.title("Hierarchical Clustering")
plt.show()

```

**Explanation**

- **k-Means** divides the data into clusters by minimizing within-cluster variance.
- **Hierarchical Clustering** builds a hierarchy of clusters and does not require a predefined \( k \).