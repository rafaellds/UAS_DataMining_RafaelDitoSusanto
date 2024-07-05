# Impor pustaka yang diperlukan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Memuat dataset
# Untuk contoh ini, kita menggunakan dataset Iris
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Memeriksa beberapa baris pertama dari dataframe
df.head()

# Menyiapkan data untuk clustering
X = df.drop('target', axis=1)

# Menerapkan clustering K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Menambahkan label cluster ke dataframe
df['cluster'] = kmeans.labels_

# Visualisasi cluster
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['cluster'], cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering of Iris Dataset')
plt.show()
