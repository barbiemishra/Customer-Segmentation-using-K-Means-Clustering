import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data 

DATA_PATH = "Mall_Customers.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    print(f"'{DATA_PATH}' not found — generating synthetic data instead.\n")
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Gender": rng.choice(["Male", "Female"], size=n),
        "Age": rng.integers(18, 70, size=n),
        "Annual Income (k$)": rng.integers(15, 140, size=n),
        "Spending Score (1-100)": rng.integers(1, 100, size=n),
    })

print("First 5 rows:")
print(df.head(), "\n")

# Select the features that represent "purchase history", Income, Spending Score 

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find the optimal K using the Elbow Method

inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(K_range)
plt.grid(alpha=0.3)
plt.savefig("elbow_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved elbow_plot.png — inspect it to confirm K.\n")

# Fit the final model

OPTIMAL_K = 5

kmeans = KMeans(n_clusters=OPTIMAL_K, init="k-means++", random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"Cluster sizes:\n{df['Cluster'].value_counts().sort_index()}\n")

# Visualize the clusters

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=80,
)

# Plot centroids back in original units

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_original[:, 0],
    centroids_original[:, 1],
    s=250,
    c="black",
    marker="X",
    label="Centroids",
)

plt.title(f"Customer Segments (K={OPTIMAL_K})")
plt.legend()
plt.savefig("customer_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved customer_clusters.png\n")

# Profile each cluster 

profile = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()
profile["Count"] = df["Cluster"].value_counts().sort_index()
print("Cluster profiles (avg income / avg spending score):")
print(profile.round(1))

# Save the labeled dataset

df.to_csv("customers_with_clusters.csv", index=False)
print("\nSaved customers_with_clusters.csv with cluster labels.")
