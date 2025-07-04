# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Load Dataset
df = pd.read_csv("E:\Mall_Customers.csv")
print(df.head())

# 2. Select Features (we'll use Annual Income and Spending Score for clustering)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. (Optional) Visualize using PCA for 2D (already 2D — skip)

# 5. Use Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 6. Fit K-Means with optimal K (let's pick K=5 based on typical elbow curves)
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 7. Visualize Clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', label='Centroids', marker='X')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

# 8. Evaluate using Silhouette Score
score = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {score:.2f}')
