import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
rfm = pd.read_csv("Dataset\online_retail_cleaned.csv")
rfm.info()
print(rfm.describe())