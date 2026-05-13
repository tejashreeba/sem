import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")
df['age']
df['age'].mean()
df['age'].median()
df['age'].std()
df['age'].min()
df['age'].max()
df['age'].max()

df['income'].unique()
df['income'].nunique()

df.groupby('income')['age'].mean()
df.groupby('income')['age'].median()
df.groupby('income')['age'].std()
df.groupby('income')['age'].std()

df.groupby(['income','age']).count()
df.groupby(['income','age']).min()
df.groupby(['income','age']).max()





