#Creating a model that will predict whether or not users will 
#click on an ad based off features of that user using fake advertising data set

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#Loading the data
ad_data=pd.read_csv("advertising.csv")

ad_data.head()

ad_data.info()

ad_data.describe()

ad_data['Age'].hist(bins=30)

#Showing area income vs age
sns.jointplot(x='Age',y='Area Income',data=ad_data,kind='scatter')

#Show time spent on site vs age
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,kind='kde')

#Showing time spent on site vs internet usage 
sns.jointplot(x='Daily Internet Usage',y='Daily Time Spent on Site',data=ad_data,kind='scatter')

plt.show()
#Spliting the data into training set and testing set using train_test_split
from sklearn.cross_validation import train_test_split

ad_data.columns

X=ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y=ad_data['Clicked on Ad']

#Splitting the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#Training and fitting a logistic regression model on the training set.
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)

#predicting values for the test data
predictions=logmodel.predict(X_test)

#Creating a report for the model
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))