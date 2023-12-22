import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes, load_iris

df=pd.DataFrame(load_diabetes().data,columns=load_diabetes().feature_names)
print(df)
df['target']=load_diabetes().target
print(df)

print(df.info())

#MACHINE LEARNING


#X = df[df.columns[:-1]] # all columns but the last one
x=load_diabetes().data
y=load_diabetes().target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model=LinearRegression()

model.fit(x_train,y_train)
predictions=model.predict(x_test)
print(y_test)
print()
print(predictions)

print(r2_score(y_test,predictions))

plt.scatter(predictions,y_test)
plt.ylabel('y test')
plt.xlabel('y prdictions')
plt.grid()
plt.show()