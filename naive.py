import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("iris.csv")

#####Split features and target
x = df.drop('species',axis = 1)
y = df['species']

######Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state=0)


#### feature scaling
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

x_train = std.fit_transform(x_train)
x_test = std.fit_transform(x_test)

####model training
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_predict = gnb.predict(x_test)

###evaluation
from sklearn.metrics import accuracy_score,precision_score,recall_score
acc = accuracy_score(y_test,y_predict)
prec = precision_score(y_test,y_predict,average='macro')
rec = recall_score(y_test,y_predict,average='macro')
acc

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test,y_predict)
cm_d = ConfusionMatrixDisplay(cm).plot()
