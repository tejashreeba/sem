import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
df = sns.load_dataset('iris')
df.info() 
df.hist() 
sns.histplot(data=df,x='sepal_length',hue='species')
plt.show() 

sns.boxplot(data=df) 
sns.boxplot(data=df,x='sepal_length') 
sns.boxplot(data=df,x='sepal_width') 
sns.boxplot(data=df,x='petal_length') 
sns.boxplot(data=df,x='petal_width') 

Q1 = df['sepal_width'].quantile(0.25)
Q3 = df['sepal_width'].quantile(0.75) 
IQR = Q3 - Q1 

lower_bound = Q1-1.5*IQR 
upper_bound = Q3+1.5*IQR 

outlier = df[(df['sepal_width']<lower_bound) | (df['sepal_width'] > upper_bound)]
print(outlier)
