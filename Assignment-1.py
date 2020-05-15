# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:51:38 2020

@author: Jinsung
"""
# Homewokr #1 :  Gaussian mixture model(GMM) classifier

# plot the fitting model
def plot_3d(classifier, data, i ,j, z):
    n = len(data)
    x = np.linspace(data['x1'].max() + .1, data['x1'].min() - .1, n)
    y = np.linspace(data['x2'].min() - .1, data['x2'].max() + .1, n)
    xx, yy = np.meshgrid(x, y)
    zz = np.array( [classifier.predict_proba(np.array([xi,yi]).reshape(1,-1))[:,1] for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] ).reshape(xx.shape)
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.gca().invert_xaxis()
    surf = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='coolwarm')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('predict')
    ax.set_title('Surface plot of Gaussian Mixture Models {} gaussian {} tol {} iteration'.format(i ,j, z))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_gaussianmixture(classifier, data, i, j, z):
    n = len(data)
    x = np.linspace(data['x1'].min() - .1, data['x1'].max() + .1, n)
    y = np.linspace(data['x2'].min() - .1, data['x2'].max() + .1, n)
    xx, yy = np.meshgrid(x, y)
    zz = np.array([classifier.predict(np.array([xi,yi]).reshape(1,-1)) for xi, yi in zip(np.ravel(xx), np.ravel(yy))]).reshape(xx.shape)
    pi = classifier.predict_proba(data)
    plt.scatter(data['x1'], data['x2'], s=50, linewidth=1, edgecolors="b", cmap=plt.cm.binary, c=pi[:, 0])
    plt.contourf(xx, yy, zz, cmap='viridis', alpha=.2)
    #plt.contourf(xx, yy, zz, alpha=0.05, cmp='jet')
    plt.title("{} gaussian {} tol {} iteration".format(i ,j, z))
    plt.show()

def predict_label(classifier0, classifier1, data, num):
    dis_4_0 = classifier0.predict_proba(data).max(axis=1)
    dis_4_1 = classifier1.predict_proba(data).max(axis=1)

    predict = []

    for i in range(len(data)):
        if dis_4_0[i] >= dis_4_1[i]:
            predict.append(0)
        else:
            predict.append(1)

    return predict

def accuracy(predict, label, num):
    real_label = label['y'].tolist()
    total = 0
    for i in range(num):
        if predict[i] == real_label[i]:
            total += 1
        else:
            total += 0
    return (total/num)*100

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

sns.scatterplot(x='x1', y='x2', hue='y', s=100, data=p1_train)
plt.title('plot for p1 train data')
plt.show()
sns.scatterplot(x='x1', y='x2', s=100, data=train_4_0)
plt.title('plot for p1 train data_label 0')
plt.show()
sns.scatterplot(x='x1', y='x2', s=100, color='orange',data=train_4_1)
plt.title('plot for p1 train data_label 1')
plt.show()

sns.scatterplot(x='x1', y='x2', hue='y', s=100, data=p2_train)
plt.title('plot for p2 train data')
plt.show()
sns.scatterplot(x='x1', y='x2', s=100, data=train_4_0)
plt.title('plot for p2 train data_label 0')
plt.show()
sns.scatterplot(x='x1', y='x2', s=100, color='orange',data=train_4_1)
plt.title('plot for p2 train data_label 1')
plt.show()


# training
n_components = [2, 4, 6 ,8, 10]
tol = [1e-3, 1e-5, 1e-9]
max_iter = [10, 20, 30, 40, 50, 60, 70, 80, 101]
max_iter = [1,2,3,4,5,6,7,8,9,10, 20, 30, 40, 50, 60, 70, 80, 101]
acc_train = []
acc_test = []

# GMM models for dataset p1
for i in n_components:
    for j in tol:
        for z in max_iter:
            print('n_componetns:{}, tol:{}, max_iter:{}'.format(i,j,z))
            model_0 = GaussianMixture(n_components=i, init_params='random', random_state=0, tol=j, max_iter=z)
            model_1 = GaussianMixture(n_components=i, init_params='random', random_state=0, tol=j, max_iter=z)
            model_0.fit(train_4_0)
            model_1.fit(train_4_1)

            plot_gaussianmixture(model_0, train_4_0, i, j, z)
            plot_3d(model_0, train_4_0, i, j, z)
            plot_gaussianmixture(model_1, train_4_1, i, j, z)
            plot_3d(model_1, train_4_1, i, j, z)

            predict_train = predict_label(model_0, model_1, p1_train_input, len(p1_train_input))
            predict_test = predict_label(model_0, model_1, p1_test_input, len(p1_test_input))
            accuracy_train = accuracy(predict_train, p1_train_label, len(predict_train))
            accuracy_test = accuracy(predict_test, p1_test_label, len(predict_train))
            acc_train.append(accuracy_train)
            acc_test.append(accuracy_test)

# GMM models for dataset p2
acc_train_p2 = []
acc_test_p2 = []

for i in n_components:
    for j in tol:
        for z in max_iter:
            print('n_componetns:{}, tol:{}, max_iter:{}'.format(i,j,z))
            model_0 = GaussianMixture(n_components=i, init_params='random', random_state=0, tol=j, max_iter=z)
            model_1 = GaussianMixture(n_components=i, init_params='random', random_state=0, tol=j, max_iter=z)
            model_0.fit(train_4_0_p2)
            model_1.fit(train_4_1_p2)

            plot_gaussianmixture(model_0, train_4_0_p2, i, j, z)
            plot_3d(model_0, train_4_0_p2, i, j, z)
            plot_gaussianmixture(model_1, train_4_1_p2, i, j, z)
            plot_3d(model_1, train_4_1_p2, i, j, z)

            predict_train = predict_label(model_0, model_1, p2_train_input, len(p2_train_input))
            predict_test = predict_label(model_0, model_1, p2_test_input, len(p2_test_input))
            accuracy_train = accuracy(predict_train, p2_train_label, len(predict_train))
            accuracy_test = accuracy(predict_test, p2_test_label, len(predict_train))
            acc_train_p2.append(accuracy_train)
            acc_test_p2.append(accuracy_test)

conv = []
for i in range(0 ,270, 18):
    point = acc_train[i:i+18]
    point_t = acc_test[i:i+18]
    plt.plot(max_iter, point, label='train')
    plt.plot(max_iter, point_t, label='test')
    for j in range(0,19):
        if  point[j]== point[j+1] == point[j+2] == point[j+3] == point[j+4]:
            plt.axvline(x=max_iter[j], color='r', linestyle='--', linewidth=2)
            conv.append(max_iter[j])
            break
    plt.legend()
    plt.ylabel('Acc')
    plt.xlabel('Iter')
    plt.show()


maximum = []
for i in range(0 ,270, 18):
    point_t = acc_test[i:i+18]
    print(point_t)
    num = point_t.index(max(point_t))
    print(max(point_t))
    print(num)
    maximum.append(num)
print(maximum)
acc_train.index(max(acc_train[0:18]))
new = zip(max_iter,acc_train[0:18])

point= new[0:18]
point= point[0:15]
for i, j in new:
    print(i)
    print(j)

for i in range(0, 270, 18):
    print(i)
    point = acc_train[i:i+18]
    print(len(point))

'''
def plot_gaussianmixture(n):
    model_x1 = GaussianMixture(n_components=10, init_params='random', random_state=0, tol=1e-9, max_iter=n)
    model_x2 = GaussianMixture(n_components=10, init_params='random', random_state=0, tol=1e-9, max_iter=n)
    model_x1.fit(p1_train_input['x1'])
    model_x2.fit(p1_train_input['x2'])
    pi = model.predict_proba(p1_train_input)
    line = model.get_params()
    plot_decision_boundary(model, p1_train_input, p1_test_label)
    plt.scatter(p1_train_input['x1'], p1_train_input['x2'], s=50, linewidth=1, edgecolors="b", cmap=plt.cm.binary, c=pi[:, 0])
    plt.title("iteration: {}".format(n))


def plot_decision_boundary(classifier, X, y, N = 10 , ax = None ):
    x_min, x_max = X['x1'].min() - .1, X['x1'].max() + .1
    y_min, y_max = X['x2'].min() - .1, X['x2'].max() + .1
    xx, yy = np.meshgrid( np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    classes = len(np.unique(y))
    
    #Check what methods are available
    if hasattr(classifier, "predict"):
        zz = np.array( [classifier.predict(np.array([xi,yi]).reshape(1,-1)) for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )
    elif hasattr(classifier, "predict_proba"):
        zz = np.array( [classifier.predict_proba(np.array([xi,yi]).reshape(1,-1))[:,1] for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )
    else :
        zz = np.array( [classifier(np.array([xi,yi]).reshape(1,-1)) for  xi, yi in zip(np.ravel(xx), np.ravel(yy)) ] )
            
    # reshape result and plot
    Z = zz.reshape(xx.shape)
    
    
    #Get current axis and plot
    if ax is None:
        ax = plt.gca()
    ax.contourf(xx, yy, Z, classes-1, cmap='viridis', alpha=.2)
    ax.scatter(X['x1'],X['x2'],c = classifier.predict(X), cmap = 'viridis')
    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
'''