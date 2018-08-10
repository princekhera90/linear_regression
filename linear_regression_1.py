# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:44:06 2018

@author: prince khera
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

def dataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        else:
            val-=step
    xs = [i for i in range(hm)]
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)

            
    

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,5.5,6,7,8,9],dtype=np.float64)

def best_fit_slope(x,y):
    m = (mean(x)*mean(y)- mean(x*y))/(mean(x)**2-mean(x**2))
    return m

xs,ys=dataset(40,40,2,correlation='neg')

m = best_fit_slope(xs,ys)
c = mean(ys) - m*mean(xs)

plt.scatter(xs,ys)
plt.plot(xs,m*xs+c)

r = 1 - ((sum(((m*xs+c)-ys)**2))/(sum(((m*xs+c)-mean(ys))**2)))


