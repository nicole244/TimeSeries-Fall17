# put your code and write up here


import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import time
import statsmodels.api as sm
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.api import qqplot

#import data
dataSource = os.path.join(os.getcwd(),"daily-minimum-temperatures-in-me.csv")
tempsData = pd.read_csv(dataSource) 


"""
Data was cleaned as there were ? characters.  
"""
#clean data
tempsData.columns = ['date','minTemp']
tempsData.date = tempsData.date.astype('datetime64')
tempsData.minTemp = tempsData.minTemp.str.replace('?','')
tempsData.minTemp = tempsData.minTemp.astype('float')
tempsData.index = tempsData.date
del tempsData['date']

#plot time series
tempsData.plot(figsize=(12,8))
plt.show()

#plot ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(tempsData.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(tempsData, lags=40, ax=ax2)
plt.show()

#fit some models based on what we saw in PACF and see how they do
arma_mod70 = sm.tsa.ARMA(tempsData, (7,0)).fit(disp=False)
print(arma_mod70.params)
print(arma_mod70.aic,arma_mod70.bic,arma_mod70.hqic)

arma_mod80 = sm.tsa.ARMA(tempsData, (8,0)).fit(disp=False)
print(arma_mod80.params)
print(arma_mod80.aic,arma_mod80.bic,arma_mod80.hqic)


print(arma_mod70.arroots)
print(arma_mod80.arroots)

print(sm.stats.durbin_watson(arma_mod70.resid.values))
print(sm.stats.durbin_watson(arma_mod80.resid.values))

"""
AR(7) appeared to be the most accurate model.  AR(8) had an increase of error.
"""


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod70.resid.plot(ax=ax);
plt.show()

#Check to see if the residuals of the AR(7) model. 
resid = arma_mod70.resid
print(stats.normaltest(resid))


fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.show()

#plot ACF and PACF of residuals
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
plt.show()

r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))


#Attempt predictions
predict_temps = arma_mod70.predict('1990-01-01', '1990-12-31', dynamic=True)
print(predict_temps)

fig, ax = plt.subplots(figsize=(12, 8))
ax = tempsData.ix['1981':].plot(ax=ax)
fig = arma_mod70.plot_predict('1990-01-01', '1990-12-31', dynamic=True, ax=ax, plot_insample=False)
plt.show()

def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

print(mean_forecast_err(tempsData.minTemp, predict_temps))

#bin data on month 
monthlyTemps = pd.DataFrame()
monthlyTemps['minTemp'] = tempsData.minTemp.resample('M').mean()

#plot monthly time series
monthlyTemps.plot(figsize=(12,8))
plt.show()

#plot ACF and PACF of monthly time series
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(monthlyTemps.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(monthlyTemps, lags=40, ax=ax2)
plt.show()

#fit an AR(3) and AR(4) model based on what was seen in the PACF plot 
arma_mod30 = sm.tsa.ARMA(monthlyTemps, (3,0)).fit(disp=False)
print(arma_mod30.params)
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)

arma_mod40 = sm.tsa.ARMA(monthlyTemps, (4,0)).fit(disp=False)
print(arma_mod40.params)
print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)

#make predictions using the AR(3) model
monthly_predict = arma_mod30.predict('1990-01-31', '1990-12-31', dynamic=True)
print(monthly_predict)
fig, ax = plt.subplots(figsize=(12, 8))
ax = monthlyTemps.ix['1981':].plot(ax=ax)
fig = arma_mod30.plot_predict('1990-01-31', '1990-12-31', dynamic=True, ax=ax, plot_insample=False)
plt.show()

def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

print(mean_forecast_err(monthlyTemps.minTemp, monthly_predict))

"""
For Melburn Temperatures between 1981 and 1990, we initally started with a
binning method of 1 week to supress the noise generated by day.  The inital 
model prediction, which utilized an AR(3), was not effective at tracking due to 
spurious noise.  We then choise to increase to bin to monthy which reduced the 
noise further.  The model now tracks the series more closely. 
"""
