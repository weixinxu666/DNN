#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from sklearn import datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.75)

# 每行数据4个特征，都是real-value的
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# 构建一个DNN分类器，3层，其中每个隐含层的节点数量分别为10，20，10，目标的分类3个，并且指定了保存位置
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="./iris_model")

# 指定数据，以及训练的步数
classifier.fit(x=X_train,
               y=Y_train,
               steps=2000)

# 模型评估
accuracy_score = classifier.evaluate(x=X_test,
                                     y=Y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

from sklearn.svm import SVC

# # 直接创建数据来进行预测
# new_samples = np.array(
#     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
# y = classifier.predict(new_samples)
# print(y)

