import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Name':['A','B','C','D','E'],
    'Gender':['Male','Female','Male','F','M'],
    'Maths':[85,90,None,120,70],
    'Science':[78,88,92,None,65],
    'Attendance':[90,95,85,300,80]
}

df = pd.DataFrame(data)

print(df)
df.isnull().sum()

df['Maths'] = df['Maths'].fillna(df['Maths'].mean())
df['Science'] = df['Science'].fillna(df['Science'].mean())

df['Gender'].unique()
df['Gender'] = df['Gender'].replace({
    'M':'Male',
    'F':'Female'
})

Q1 = df['Maths'].quantile(0.25)
Q3 = df['Maths'].quantile(0.75)
IQR = Q3-Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3+1.5*IQR
outliers = df[(df['Maths'] < lower_bound) | (df['Maths'] > upper_bound)]
outliers

sns.histplot(data=df,x='Attendance')
plt.show()
df['Attendance_log'] = np.log(df['Attendance'])

sns.histplot(data=df,x='Attendance_log')
plt.show()













