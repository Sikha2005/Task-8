# Task-8
# Mall Customer Segmentation with K-Means Clustering 🛍️

## 📌 Objective
This project performs **unsupervised machine learning** using the **K-Means Clustering algorithm** to segment customers based on their **Annual Income** and **Spending Score** from a mall customer dataset.

---

## 📂 Dataset
**Filename:** `Mall_Customers.csv`

**Columns:**
- `CustomerID`
- `Genre`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

---

## 🚀 Tasks Performed

1. **Load and visualize the dataset**
2. **Standardize the features** for fair clustering
3. **Use the Elbow Method** to determine the optimal number of clusters
4. **Fit K-Means Clustering** with the optimal K value
5. **Visualize the clusters** with color-coded scatter plots
6. **Evaluate clustering quality** using the **Silhouette Score**

---

## 📊 Visualizations

- **Elbow Curve**: Helps identify the optimal number of clusters (K)
- **Cluster Scatter Plot**: Visualizes customer segments with cluster centers

---

## 📈 Evaluation Metrics

- **Inertia** (within-cluster sum of squares)
- **Silhouette Score** (measure of cluster cohesion and separation)

---

## 🛠️ Tools & Libraries

- Python 3.x
- `pandas`
- `scikit-learn`
- `matplotlib`

---
-----

## 📊 Sample Output

When you run the program, the following output appears:

```text
   CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40

Silhouette Score: 0.55
