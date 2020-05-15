# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:24:47 2020

@author: Jinsung
"""

# overfitting을 피해 optimal한 모델을 생성하자.
# 언제 emerge가 빨리 될지 판단하는 걸 수도있음


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.svm import SVC
import argparse

class NN(nn.Module):
    def __init__(self, activation):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(2,30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30,2)

        if activation == 'sigmoid':
            self.a = F.sigmoid()
        elif activation == 'relu':
            self.a = F.relu()
        elif activation == 'tanh':
            self.a = F.tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.a(x)
        x = self.fc2(x)
        x = self.a(x)
        x = self.fc3(x)
        x = self.a(x)
        return x


def accuracy(predict, label, num):
    real_label = label['y'].tolist()
    total = 0
    for i in range(num):
        if predict[i] == real_label[i]:
            total += 1
        else:
            total += 0
    return (total/num)*100

parser = argparse.ArgumentParser()
parser.add_argument('--activation', default='sigmoid')
parser.add_argument('--optimizer', default='SGD')
args = parser.parse_args()

# read data
p1_train_input = pd.read_csv("p1_train_input.txt", names=['x1','x2'], sep='\s+')
p1_train_label = pd.read_csv("p1_train_target.txt", names=['y'], sep='\s+')
p1_test_input = pd.read_csv("p1_test_input.txt", names=['x1','x2'], sep='\s+')
p1_test_label = pd.read_csv("p1_test_target.txt", names=['y'], sep='\s+')
p1_train = pd.concat([p1_train_input, p1_train_label], axis=1)
p1_test = pd.concat([p1_test_input, p1_test_label], axis=1)


p2_train_input = pd.read_csv("p2_train_input.txt", names=['x1','x2'], sep='\s+')
p2_train_label = pd.read_csv("p2_train_target.txt", names=['y'], sep='\s+')
p2_test_input = pd.read_csv("p2_test_input.txt", names=['x1','x2'], sep='\s+')
p2_test_label = pd.read_csv("p2_test_target.txt", names=['y'], sep='\s+')
p2_train = pd.concat([p2_train_input, p2_train_label], axis=1)
p2_test = pd.concat([p2_test_input, p2_test_label], axis=1)

#plot the y = 0, 1
train_4_1 = p1_train[p1_train['y'] == 1].drop(['y'], axis=1)
train_4_0 = p1_train[p1_train['y'] == 0].drop(['y'], axis=1)

train_4_1_p2 = p2_train[p2_train['y'] == 1].drop(['y'], axis=1)
train_4_0_p2 = p2_train[p2_train['y'] == 0].drop(['y'], axis=1)

criterion = nn.CrossEntropyLoss()

if args.optimizer == 'SGD':
    optimizer = optim.SGD(NN.parameters(), lr=0.01, momentum=0.9)
elif args.optimizer == 'ADAM':
    optimizer = optim.Adam(NN.parameters(), lr=0.01)

net = NN(args.activation)


mlp = MLPClassifier(hidden_layer_sizes=[30,30,30], learning_rate_init=0.01, verbose=True, max_iter=1000)
mlp.fit(p1_train_input, p1_train_label)
mlp.predict(p1_train_input)

accuracy(mlp.predict(p1_train_input), p1_train_label, len(p1_train_label))

accuracy(mlp.predict(p1_test_input), p1_test_label, len(p1_test_label))


labels = np.ravel(p1_train_label)
model = SVC(kernel='rbf', C=10000000)
model.fit(p1_train_input, labels)
model.support_vectors_

print("훈련 세트 정확도: {:.2f}".format(model.score(p1_train_input, p1_train_label)))
print("테스트 세트 정확도: {:.2f}".format(model.score(p1_test_input, p1_test_label)))


plt.scatter(p1_train_input['x1'],p1_train_input['x2'],c=labels, s=30, cmap=plt.cm.Paired)


# 초평면(Hyper-Plane) 표현
ax = plt.gca()
 
xlim = ax.get_xlim()
ylim = ax.get_ylim()
 
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)
 
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])
 
# 지지벡터(Support Vector) 표현
ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=60, facecolors='r')
 
plt.show()