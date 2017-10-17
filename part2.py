# put your code and discussion here


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
dataSource = os.path.join(os.getcwd(),"internationalAirlinePassengers.csv")
airplaneData = pd.read_csv(dataSource)


airplaneData.columns = ['Month','Value']
airplaneData['Month'] = airplaneData.Month.astype('datetime64')
airplaneData.index = airplaneData.Month
del airplaneData['Month']


#plot time series
airplaneData.plot(figsize=(12,8))
plt.show()

#plot ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(airplaneData.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(airplaneData, lags=40, ax=ax2)
plt.show()

airplaneData = airplaneData.apply(np.log)

airplaneData.plot(figsize=(12,8))
plt.show()


detrend = sm.tsa.tsatools.detrend(airplaneData,order=2)
dtData = pd.DataFrame(detrend)
dtData.index = airplaneData.index
dtData.columns = ['values']

dtData.plot(figsize=(12,8))
plt.show()

#plot ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dtData.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dtData, lags=40, ax=ax2)
plt.show()

#fit some models based on what we saw in PACF and see how they do
#arma_mod110 = sm.tsa.ARMA(dtData, (11,0)).fit(disp=False)
#print(arma_mod110.params)
#print(arma_mod110.aic,arma_mod110.bic,arma_mod110.hqic)

arma_mod130 = sm.tsa.ARMA(dtData, (13,0)).fit(disp=False)
print(arma_mod130.params)
print(arma_mod130.aic,arma_mod130.bic,arma_mod130.hqic)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax = arma_mod130.resid.plot(ax=ax);
plt.show()

#Check to see if the residuals of the AR(7) model. 
resid = arma_mod130.resid
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



#Attempt predictions
predict_temps = arma_mod130.predict('1959-01-01', '1960-12-01', dynamic=True)
print(predict_temps)

fig, ax = plt.subplots(figsize=(12, 8))
ax = dtData.ix['1950':].plot(ax=ax)
fig = arma_mod130.plot_predict('1959-01-01', '1960-12-01', dynamic=True, ax=ax, plot_insample=False)
plt.show()

#def mean_forecast_err(y, yhat):
#    return y.sub(yhat).mean()

#print(mean_forecast_err(dtData.values, predict_temps))