import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

# Display first rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Initial statistics
print(df.describe())

# Variable information
print(df.info())

# Dataset dimensions
print(df.shape)

# Datatypes
print(df.dtypes)

# Fill missing age values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Type conversion
df['Age'] = df['Age'].astype(int)

# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['age','fare']] = scaler.fit_transform(df[['age','fare']])

# Convert categorical variable
df['Sex'] = df['Sex'].map({
    'male':0,
    'female':1
})

print(df.head())
