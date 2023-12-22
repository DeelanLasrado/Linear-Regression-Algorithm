import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


var_x = np.array([1.1,1.3,1.5,2.0,2.2,2.9,3.0, 3.2, 3.2, 3.7,3.9,4.0, 4.0, 4.1,4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3,10.5, 6.8, 7])
var_y = np.array([39343, 46205,37731, 43535, 39821, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 73504, 79123, 83088, 81363, 93940, 91738, 98217, 101302, 113812, 109431, 105582, 116969, 12635, 122391, 121872])

X=  var_x.reshape(-1, 1)
y = var_y
plt.scatter(var_x,var_y)
#plt.show()

df = pd.DataFrame({"Experience":var_x,"Salary":var_y})
df.head()

#X = df.Experience
#y = df.Salary


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_test)
print()
print(y_pred.round(2 ))

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))


#accuracy
from sklearn.metrics import r2_score
print(model.score(X_test,y_test))

print(r2_score(y_test,y_pred))