#-*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function

# 引入各项扩展库
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def read(path):
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_data=pd.read_csv(path,names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
    dataset=raw_data.copy()
    dataset=dataset.dropna()
    origin=dataset.pop('Origin')
    dataset['usa']=(origin==1)*1.0
    dataset['Europe']=(origin==2)*1.0
    dataset['Japan']=(origin==3)*1.0
    train_dataset=dataset.sample(frac=0.8, random_state=0)
    test_dataset=dataset.drop(train_dataset.index)
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')
    normed_train_data = norm(train_dataset,train_stats['mean'],train_stats['std'])
    normed_test_data = norm(test_dataset,train_stats['mean'],train_stats['std'])

    model = LinearRegression()
    model.fit(normed_train_data, train_labels)  # 训练
    prediction = model.predict(normed_test_data)  # 预测
    print(prediction)  # 显示预测值
    print(test_labels)  # 显示真实值

    mse = mean_squared_error(test_labels,prediction)
    mse = mse **(0.5)#一般再开个根号
    print(mse)


def norm(x,y,z):
    return (x - y) / z

if __name__ == '__main__':
    path='D:\\ML\\linear_regression\\auto-mpg.data-original'
    read(path)