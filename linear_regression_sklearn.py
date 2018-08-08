# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:12:07 2018

@author: prince khera
"""

import pandas as pd
import quandl, datetime
import math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')#to make it look descent

df = quandl.get('WIKI/GOOGL',authtoken="zPZHfHEWrExsRmgqnDzE")

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'Adj. Volume','HL_PCT','PCT_change']]
df.fillna(-9999,inplace=True)

forecast_col = 'Adj. Close'

forecast_out=int(math.ceil(0.5*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))#1 for axis
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
#pickling = serialization of any python object
#with open('linearregression.pickle','wb') as f:
#   pickle.dump(clf,f)
    
#pickle_in=open('linearregression.pickle','rb')
#clf = pickle.load(pickle_in)
accuracy = clf.score(X_test,y_test)#squared error

#clf_s = svm.SVR()
#clf_s.fit(X_train,y_train)
#accuracy_s = clf.score(X_test,y_test)#squared error

forecast_set = clf.predict(X_lately)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name#returns a timestamp
last_unix = last_date.timestamp()#digital timestamp
one_day = 86400#seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:#including forecast set in df
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for j in range(len(df.columns)-1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')

    




















