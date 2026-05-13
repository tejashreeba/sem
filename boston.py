import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("housing_data.csv")
df.dropna(inplace=True)

x=df.drop('MEDV',axis=1)
y=df['MEDV']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)

plt.figure(figsize = (10,8))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
plt.title("Correlation index")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test,y_predict)
plt.xlabel("Actual Values")
plt.ylabel("predicted values")
plt.show()





