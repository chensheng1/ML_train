#-*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report

iris_path = 'D://ML//LogisticRegression//iris.csv'
data = pd.read_csv(iris_path)
x=data.iloc[:,[1,2,3,4]].values
y=data.iloc[:,[5]].values
y=np.where(y=='setosa',1,0)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#数据标准化
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)


logreg = linear_model.LogisticRegression(C=1000.0, random_state=0,solver='lbfgs',multi_class='multinomial')
logreg.fit(x_train_std, y_train)

prepro = logreg.predict(x_test_std)   #返回属于某一类别
#prepro=logreg.predict(x_test_std)  #返回属于某一类别的概率

print(classification_report(y_test, prepro))   #类别的区分评估
print(prepro)
print(logreg.score(x_test_std,y_test))
