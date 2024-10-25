#%%
# Import library
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

#%%
# Import data
dataset = pd.read_excel("tech_layoffs.xlsx")

# Mendeteksi dan menghapus NaN
print("Jumlah NaN sebelum penghapusan:")
print(dataset.isna().sum())
dataset = dataset.dropna()
print("\nJumlah NaN setelah penghapusan:")
print(dataset.isna().sum())

# Mengambil variabel untuk clustering
X = dataset.iloc[:, [5,7]].values

#%%
# Metode Elbow untuk Menentukan Jumlah Optimal Cluster
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Metode Elbow')
plt.xlabel('Jumlah Clusters')
plt.ylabel('WCSS')
plt.show()

#%%
# Menerapkan KMeans dengan Jumlah Cluster yang Optimal
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

#%%
# Visualisasi Hasil Clustering
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=50, c='magenta', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, c='yellow', label='Centroids')
plt.title('Cluster Harga Saham')
plt.xlabel('Harga Saham Pasar Buka')
plt.ylabel('Harga Saham Pasar Tutup')
plt.legend()
plt.show()

# %%
