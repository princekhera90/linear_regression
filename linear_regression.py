a# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:52:48 2018

@author: prince khera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalization(X,Y):
    for i in range(1,len(X)):
        M = max(X[i])
        m = min(X[i])
        X[i] = (X[i]-m)/(M-m)
    M = np.amax(Y)
    m = np.amin(Y)
    Y = (Y-m)/(M-m)
    return X,Y

def cost(X,Y,t):
    y = t.T.dot(X)
    D = (y-Y)
    d = D**2
    J = d.sum()/(2*max(Y.shape))
    return J

def gradient_descent(X,Y,t,ne,a):
    Jh = []
    for i in range(ne):
        y = t.T.dot(X)
        D = (y-Y)
        t = t - a*((X.dot(D.T))/max(Y.shape))
        J = cost(X,Y,t)
        Jh.append(J)
    return t,Jh
    
def normal_eq(X,Y):
    s = np.linalg.inv(X.dot(X.T))
    s1 = X.dot(Y.T)
    S = s.dot(s1)
    return S
#import csv
file = 'ex1data1.txt'
df = pd.read_csv(file,names=['a','b'])
#dataset = []
#with open(file,'r') as f:
    #data = csv.reader(f)
    #for d in data:
        #dataset.append(d)
n = len(df.columns) - 1  
m = len(df)      
X = np.array([df['a']])
Y = np.array([df['b']])
#plt.plot(X,Y)
#plt.show()
x = np.ones((1,m))
X = np.concatenate((x,X))
t = np.zeros((n+1,1))

t,jh = gradient_descent(X,Y,t,100000,.003)
y_pred = t[0] + t[1]*X[1]
plt.scatter(X[-1].reshape(1,97),Y)
plt.plot(X[-1],y_pred,'r')

s = normal_eq(X,Y)




  