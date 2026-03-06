import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
# Load dataset
rfm = pd.read_csv("Dataset\online_retail_cleaned.csv")
rfm.info()
print(rfm.describe())
rfm['InvoiceDate'] = pd.to_datetime(rfm['InvoiceDate'])
# Diperlukan merubah kembali ke datetime karena sebelumnya sudah diubah ke string 
# saat menyimpan ke CSV.
reference_date = rfm['InvoiceDate'].max()
rfm = rfm.groupby('CustomerID').agg(
    last_purchase = ('InvoiceDate', 'max'),
    Frequency     = ('InvoiceNo', 'nunique'),
    Monetary      = ('Revenue', 'sum')  
)
rfm['Recency'] = (reference_date - rfm['last_purchase']).dt.days  

# Scale data
rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features) # Standarisasi data RFM
# tidak perlu menggunakan kutip karena variabel rfm_scaled sudah berupa array numpy.

#elbow plot untuk menentukan jumlah cluster optimal

# Elbow Plot
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Cluster')
plt.ylabel('WCSS Value')
plt.show()


print(f"{'k':<5} {'Silhouette':>12} {'DBI':>10}")
print("-" * 30)
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(rfm_scaled)
    sil = silhouette_score(rfm_scaled, km.labels_)
    dbi = davies_bouldin_score(rfm_scaled, km.labels_)
    print(f"k={k:<3} Silhouette={sil:.4f}  DBI={dbi:.4f}")

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  
kmeans.fit(rfm_scaled)                          
rfm['cluster_final'] = kmeans.labels_
print(rfm.groupby('cluster_final')[['Recency', 'Frequency', 'Monetary']].mean())

# simpan hasil clustering ke file CSV lama
rfm[['Recency', 'Frequency', 'Monetary', 'cluster_final']].to_csv("Dataset\online_retail_rfm.csv")