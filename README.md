# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values.

## Program:
```python
'''
Program to implement the Logistic Regression Model to Predict the Placement Status of Student.
Developed by : K SANTHAN KUMAR
RegisterNumber : 212223240065
'''
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"]= le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
c_report = classification_report(y_test,y_pred)
c_report

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### Head 
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/6a256453-451f-4171-935a-1db371002b68)
### After removing sl_no , salary
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/d8680a74-fcac-486b-b508-92761e5ec9e1)
### Null data 
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/a7dbeeea-8de8-46a7-bf20-f59980fbd9f5)
### Duplicated sum
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/c7fb3459-228a-4aab-a6ef-0edc6923ef42)
### Label Encoder
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/659547bc-8cfe-4d89-b906-169e516c3645)
### After removing the last column
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/3ae6ad4d-7629-4d51-b6f6-bf90f174d587)
### Displaying the status
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/78a45b96-783a-42c4-956d-74b01612de73)
### Prediction of y
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/5effafab-09db-4613-9102-e34812e71e71)
### Accuracy score
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/d21a9da9-5df9-4f77-948d-21e545edd318)
### Confusion 
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/4c888ccd-e65d-43df-9fcd-0ef304b2e376)
### Classification report
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/b0db5fc0-87ac-4448-a893-6ff1a2d51100)
### Prediction
![image](https://github.com/Madhavareddy09/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145742470/35b841aa-992a-4eda-a43c-4f982e82daa1)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
