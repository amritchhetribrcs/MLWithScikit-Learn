import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 2(pow)
y = np.dot(X, np.array([1, 2])) + 2**X[3][0]
# y_1 = 1 * 1 + 2 * 1 + 4=7
# y_2 = 1 * 1 + 2 * 2 + 4=9
# y_3= 2+4+4=10
#y_4= 2+6+4=12
print("X", X)
print("Y", y)
# Training Model with fit() function!
reg = LinearRegression().fit(X, y)
print("Training Result:",reg)
# Checking accuracy
print("Scores", reg.score(X, y))
print(reg.coef_)
# Checking Intercept  of function
print(reg.intercept_)
print("Predicting Result:", reg.predict(np.array([[3, 5]])))
