#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA

# 加载数据
def load_data(): 
    print("====>>===>>===>> LoadData")
    trainData = pd.read_csv('data/train.csv')
    testData = pd.read_csv('data/test.csv')
    x_train = trainData.values[:, 1:]
    y_train = trainData.values[:, 0]
    x_test = testData.values[:, :]
    # 归一化
    x_train = x_train/255
    x_test = x_test/255
    return x_train, y_train, x_test
    pass    

# 降低数据维度
def data_pca(x_train, x_test, COMPONENT_NUM):
    print("====>>===>>===>> PCA ")
    pca = PCA(n_components=COMPONENT_NUM, copy=True, whiten=False)  # 创建一个 PCA 对象
    pca.fit(x_train)    # 构建 PCA 模型
    pcaXTrain = pca.transform(x_train)
    pcaXTest = pca.transform(x_test)

    return pcaXTrain, pcaXTest

# 训练模型    
def create_model(x_train, y_train, x_test):
    print("====>>===>>===>> TrainModel ")
    # 定义模型的时候注意初始化输入形状，避免保存的模型有问题
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=500, input_shape=(x_train.shape[1], ), activation='relu'))
    model.add(keras.layers.Dense(units=500, activation='relu'))
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

    # epochs：表示一共训练的周期， batch_size：把多少层组合成一个训练单元，多线程加速
    history = model.fit(x_train, y_train, epochs=10, batch_size=32)
    model.summary()
    #保存模型
    model.save('model/model1.h5')
    return history

def saveResultData(x_test):
    model = keras.models.load_model('model/model1.h5')
    # 测试模型
    result = model.predict(x_test)
    print(result)

    lenth = len(result)
    test_label = np.zeros((lenth, 1))
    for i in range(lenth):
        test_label[i] = np.argmax(result[i])
    print(test_label)
    # 保存预测结果  , range(1, 28) 到不了 28 
    # 注意保存进入数据库中的格式以及数据的切片
    data = pd.DataFrame({'ImageId': range(1, lenth + 1), 'Label': test_label[:, 0]})
    data.to_csv('data/sample_submission.csv')
  

x_train, y_train, x_test = load_data()
pcax_train, pcax_test = data_pca(x_train, x_test, 0.9)
try:
    saveResultData(pcax_test)
except:
    history = create_model(pcax_train, y_train, pcax_test)
    saveResultData(pcax_test)

'''
# 单独预测某个值，要注意数组格式
In [4]: na = np.array([list(pcax_test[0])],)
In [5]: model = keras.models.load_model('model/model1.h5'')
  File "<ipython-input-5-4699a2c01a4f>", line 1
    model = keras.models.load_model('model/model1.h5'')                                                    ^
SyntaxError: EOL while scanning string literal
In [6]: model = keras.models.load_model('model/model1.h5')
In [7]: model.predict(na)
Out[7]:
array([[6.1574175e-15, 4.9995848e-17, 1.0000000e+00, 1.5179449e-17,
        3.6376498e-19, 3.7012272e-26, 5.9435442e-17, 1.0842079e-13,
        6.5080575e-19, 1.1886531e-23]], dtype=float32)
'''
 