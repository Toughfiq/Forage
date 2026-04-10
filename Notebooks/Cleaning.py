import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset and describe the data
retail = pd.read_csv("Dataset\online_retail.csv",on_bad_lines='skip', sep=";", decimal= ",", encoding="latin1"  )
print("===================== Initial Data Info =====================")
retail.info()
print("===================== Descriptive Statistics =====================")
print(retail.describe())
print("===================== Row and Column Count =====================")
print(retail.shape)    # lihat jumlah baris & kolom
print("===================== Retail columns =====================")
print(retail.columns)  # lihat nama kolom
print("===================== Data Types =====================")
print(retail.dtypes)   # tipe data tiap kolom
print ("===================== Data Head =====================")
print(retail.head())   # lihat 5 baris pertama

# =============== Cleaning Data ===============
# Ubah tipe data
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], dayfirst=True)
# Buang nilai Quantity negatif
retail = retail[retail['Quantity'] > 0] 
# Hapus baris dengan CustomerID kosong
retail = retail.dropna(subset=['CustomerID'])
# Hapus harga yang tidak valid
retail = retail[retail['UnitPrice'] > 0]
# Reset index
retail = retail.reset_index(drop=True)
# Membuat kolom revenue.
retail['Revenue'] = retail['Quantity'] * retail['UnitPrice']
print("=============== Cleaned Data Head + Add Revenue collumn ================")
print(retail[['Quantity', 'UnitPrice', 'Revenue']].head())
# Cek hasil awal
print("===================== Cleaned Data Info =====================")
print(retail.info())
print("===================== Cleaned Data Descriptive Statistics =====================")
print(retail.describe())

# Simpan DataFrame yang sudah dibersihkan ke file CSV baru
retail.to_csv("Dataset\online_retail_cleaned.csv", index=False)