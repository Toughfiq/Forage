import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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
rfm_scaled = scaler.fit_transform(rfm_features)