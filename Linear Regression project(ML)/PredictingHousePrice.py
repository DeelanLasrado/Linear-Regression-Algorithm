import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Q: based on given question predict the house value
#boston = load_boston()
boston = fetch_california_housing()
df = pd.DataFrame(boston.data)
df.columns = boston.feature_names
#df=pd.DataFrame(boston.data,columns=boston.feature_names)
print(df.head())

'''
# Columns Informations
* CRIM per capita crime rate by town
* ZN proportion of residential land zoned for lots over 25,000 sq.ft.
* INDUS proportion of non-retail business acres per town
* CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
* NOX nitric oxides concentration (parts per 10 million)
* RM average number of rooms per dwelling
* AGE proportion of owner-occupied units built prior to 1940
* DIS weighted distances to five Boston employment centres
* RAD index of accessibility to radial highways
* TAX full-value property-tax rate per 10,000usd
* PTRATIO pupil-teacher ratio by town
* B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
* LSTAT % lower status of the population
'''

df['PRICE'] = boston.target
print(df.head())

print(df.tail())
print(df.isnull().sum())


#machine learning
X = np.array(df.drop('PRICE', axis=1))
y = np.array(df.PRICE)
#x=boston.data
#y=boston.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
                                                    random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("\n")
print(y_test)
print()
print(y_pred)

print(model.score(X_test,y_test))
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

plt.scatter(y_test,y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid()
plt.plot([min(y_test),max(y_test)],[min(y_pred),max(y_pred)], color='red')
plt.title('Actual Price V/s Predicted Price')
plt.show()



#klib library



