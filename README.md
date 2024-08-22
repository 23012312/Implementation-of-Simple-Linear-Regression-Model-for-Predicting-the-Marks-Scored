# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K.ABHINESWAR REDDY
RegisterNumber:212223040084
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
df.head()

![image](https://github.com/user-attachments/assets/07d29f44-c180-442e-9dd8-c8fa1fc5a9fa)

df.tail()

![image](https://github.com/user-attachments/assets/952943d7-d790-4439-babd-4ce1a3061500)

Array value of X

![image](https://github.com/user-attachments/assets/809b20bc-5f83-4ae9-be0d-31d01c5fa5f5)

Array value of Y

![image](https://github.com/user-attachments/assets/1d691683-4124-4500-844b-f7bbc639ee8a)

Values of Y prediction

![image](https://github.com/user-attachments/assets/f368e984-db52-4fce-aa4c-4eacdce9d6d4)

Array values of Y test

![image](https://github.com/user-attachments/assets/a36ab1cc-ee73-48ab-99cf-868a26d139bc)

Training Set Graph

![image](https://github.com/user-attachments/assets/c3b9b623-5d54-4bf0-8679-35f99e294e38)

Test Set Graph

![image](https://github.com/user-attachments/assets/75afab1f-8e6c-4065-85a7-ffafc29f0bea)

Values of MSE, MAE and RMSE

![image](https://github.com/user-attachments/assets/48ee217b-3c31-4fc8-97fd-31be2ee98fc9)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
