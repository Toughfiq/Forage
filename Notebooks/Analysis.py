import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
rfm = pd.read_csv("Dataset\online_retail_cleaned.csv")
rfm.info()
print(rfm.describe())
rfm['InvoiceDate'] = pd.to_datetime(rfm['InvoiceDate'])
reference_date = rfm['InvoiceDate'].max()
rfm = rfm.groupby('CustomerID').agg(
    last_purchase = ('InvoiceDate', 'max'),
    Frequency     = ('InvoiceNo', 'nunique'),
    Monetary      = ('Revenue', 'sum')  
)
rfm['Recency'] = (reference_date - rfm['last_purchase']).dt.days  
