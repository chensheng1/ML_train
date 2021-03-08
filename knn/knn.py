#-*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def iris_type(s):
    it = {b'setosa': 0, b'versicolor': 1, b'virginica': 2}
    return it[s]
iris_path = 'D://ML//LogisticRegression//iris.csv'
data = pd.read_csv(iris_path)
x=data.iloc[:,[1,2,3,4]].values
y=data.iloc[:,[5]].values
'''
y=data.iloc[:,[5]]
it = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
y=y['Species'].map(it).values   #类型转换
print(y)
'''
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
iris_y_predict = knn.predict(x_test)
probility=knn.predict_proba(x_test)
score=knn.score(x_test,y_test,sample_weight=None)
print(iris_y_predict)
print(y_test)
print(probility)
print(score)