import pandas as pd
from google.colab import drive
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path='//content/Household energy unit data.csv'
data=pd.read_csv(file_path)
print(data)
print(data.info())
print(data.isnull().sum())

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(data.iloc[:,-1],bins=30,kde=True)
plt.title("Distribution")
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10,6))
for i ,col in enumerate(data.columns[:3]):
  sns.boxplot(x=col,y=data.iloc[:,-1],data=data)
  plt.title("Target")
  plt.show()
