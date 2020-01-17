# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.



Homework assignmanet for Flexibility AS

 Forecasting of future power prices assignment:

This assignment is an open task where you can use Python and any open-source library you prefer.
You shall implement a straightforward model (task 1), and one more advanced
forecasting model (task 2) to forecast future power prices. The model can, e.g.
be based on your prefered choice, and you can, e.g. use statistics, regression
or ml/deep learning. The options are up to you.
If you have any question, please ask us! (we appreciate questions)

 
Task 1: Daily-average prices based on historical prices
Task 2: Hourly prices based on historical prices and hydro level

 

Process:
Make a private repro and share it with us.
Make the model in the repro. 
Read the csv files and feed the model with data, files provided in this link https://drive.google.com/open?id=1IzxeRvx8LCt_a5k2R0jTX1kUJlxyfWWv
Make comments so it is easy to read
 

Run the model for some weeks in 2019, (e.g. seven forecasts in week 50 2019, and 16.5, 17.5, 18,5, 19.5, 20.5)
· Present the forecast results and the skills of the model.
· Present the rationale behind your design choice
· Write some short line what would be your next steps to improve the model


"""


# Import standard data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from os.path import join

path_directory = 'C:\\Users\\dangr\\Desktop\\Task_Flexibility\\Nordpool_workaround_files'

# Read the files
# There are some troubles with the file format
# Due to time restraints: Workaround converting to *.csv files. 
# For a real solution, get around the read-in and decoding problem

###############################################################################
# Historic spot prices in NOK/MWh
# hourly data (weekly data only for one year available)
# The files contain the date and 1h intervals
# Different regions:
    #2013: SYS,SE1,SE2,SE3,SE4,FI,DK1,DK2,Oslo,Kr.sand,Bergen,Molde,Tr.heim,Tromsoe,EE,ELE,LV,LT

# For now, select only norwegian regions (other regions lack partially data)
col_NO = ['Oslo', 'Bergen', 'Kr.sand', 'Molde', 'Tr.heim', 'Tromsø']

# Comment: I like to program using the spyder editor and making heavily use of
# the interactive variable explorer. That's why I didn't put many print 
# statements here although it is clearly important to look at the data first.

filenames = ['elspot-prices_2013_hourly_nok.csv',
             'elspot-prices_2014_hourly_nok.csv',
             'elspot-prices_2015_hourly_nok.csv',
             'elspot-prices_2016_hourly_nok.csv',
             'elspot-prices_2017_hourly_nok.csv',
             'elspot-prices_2018_hourly_nok.csv',
             'elspot-prices_2019_hourly_nok.csv']
data_tmp = []
for filename in filenames:
    #data = pd.read_excel(join(path_directory,filename), header=2, encoding="utf8", errors='ignore')
    data = pd.read_csv(join(path_directory,filename), header=2, index_col=0,
                       decimal=',', encoding='ANSI')
    data_tmp.append(data)
data = pd.concat(data_tmp)

# Time format in the files needs special attention
# Hours contain the local time period but follows summer/winter time
# Switch to summer time: Line with nan
# Switch to winter time: extra line with same index!

# Drop line where summer time happens
date = data.dropna(inplace=True, thresh=5)

# extract time from hours column
data['hour'] = [int(string[:2]) for string in data['Hours'].values]
data['hour_winter'] = list(np.arange(0,24))*(len(data)//24)
datetimes = [dt.datetime.strptime(s, '%d/%m/%Y') for s in data.index.values]
# Add winter time hour to it to create unique index (ignore summer time in the following)
data['date'] = [date.replace(hour=h) for date,h in zip(datetimes, data.hour_winter.values)]
data.set_index(data.date, inplace=True)

# Visualize
data[col_NO].plot()
plt.title('Power spot price in Norwegian cities in NOK/MWh')
plt.legend()

# Inspection:
# 1) Prices comparable between Norwegian cities
# 2) Daily price cycle
# 3) General trend but only weak seasonal dependency
# 4) Increase of intra-month fluctuations in the last few years
# 5) Some distinct outliers. Maybe could be linked to specific events? e.g. 1/3/2018

###############################################################################
###############################################################################
###############################################################################
# Task 1
# The power prices in Norway fluctuate hourly.
# We want you to be able to forecast the next-day power prices based on 
# historical observation of the power price.
# Forecast the average price for the next day based on historical data.

# Clean daily fluctuations by averaging
# There is still the problem with summer/winter time; ignore for now
data_daily = data[col_NO].resample('D').mean()
data_weekly = data[col_NO].resample('W').mean()

plt.figure()
plt.title('Spot prices in Oslo')
data.Oslo.plot(label='hourly')
data_daily.Oslo.plot(label='daily')
data_weekly.Oslo.plot(label='weekly')
plt.legend()

data_daily['weekday'] = data_daily.index.day_name()
# Derivative
data_daily['Oslo_diff'] = [data_daily.Oslo.values[i+1] - data_daily.Oslo.values[i] \
                         for i in range(len(data_daily)-1)] + [np.nan]
data_daily.dropna(inplace=True)

plt.figure()
data_daily.Oslo_diff.plot()
plt.title('Change in spot price in Oslo')

#Select the feature
feature = 'Oslo_diff'

# Autocorrelations
from statsmodels.tsa.stattools import acf, pacf

autocorr = acf(data_daily[feature], nlags=365)
autocorr_part = pacf(data_daily[feature], nlags=365)

plt.figure()
plt.title('Autocorrelation: ' +  feature)
plt.plot(autocorr, label='acf')
plt.plot(autocorr_part, label='pacf')
plt.axvline(7)
plt.axvline(14)
plt.xlabel('Lag in days')
plt.legend()

#Conclusion from the autocorrelation:
# Sequent fluctuations are not directly correlated 
# but there is a weekly correlation

# Seperate the time series in training and test series; roughly 4:1
train = data_daily[feature][:2000]
test = data_daily[feature][2000:]

# Time series prediction: Autoregressive Integrated Moving Averages
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

model = ARIMA(train, order=(7,1,0)) # with q=0, it basically becomes a pure AR model
model_fit = model.fit(disp=False)
model_fit.save('arima_model.pkl')
window = model_fit.k_ar
coef = model_fit.params

plt.figure()
plt.title('Training time series')
plt.plot(train, label='Real prices')
plt.plot(model_fit.fittedvalues, color='red', label='Predicition')
error_training = mean_squared_error(train[1:], model_fit.fittedvalues)
print('Training')
print('MSE: %.1f' % error_training)
print('STD: %.1f' % np.sqrt(error_training))

# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
	yhat = coef[0]
	for i in range(1, len(coef)):
		yhat += coef[i] * history[-i]
	return yhat

# Check how the model performs for the test time series
history = list(train.values) # append new predictions here
predictions = list()
for t in range(len(test)):
	next_prediction = predict(coef, history)
	obs = test[t]
	predictions.append(next_prediction)
	history.append(obs)
error = mean_squared_error(test, predictions)
print('Test')
print('MSE: %.1f' % error)
print('STD: %.1f' % np.sqrt(error))

# Visualize result
plt.figure()
plt.plot(test, label='Real price')
plt.plot(test.index, predictions, color='red', label='Predicted price')
plt.legend()
plt.show()

# Prediction for week 50
# Train the model with the full data
model_full = ARIMA(data_daily[feature], order=(7,1,0))
model_full_fit = model_full.fit(disp=False)

plt.figure()
plt.title('Full time series')
plt.plot(data_daily[feature], label='Real prices')
plt.plot(model_full_fit.fittedvalues, color='red', label='Predicition')
error_training = mean_squared_error(data_daily[feature].values[1:], model_full_fit.fittedvalues)
print('Training')
print('MSE: %.1f' % error_training)
print('STD: %.1f' % np.sqrt(error_training))

# Predict the price (not the difference)
price_diff_abs = pd.Series(model_full_fit.fittedvalues, copy=True)
price_diff_abs = price_diff_abs.cumsum()

data_daily['Oslo_predicted'] = data_daily['Oslo']+price_diff_abs

print('Week 50')
print(data_daily[['weekday','Oslo','Oslo_predicted']][-9:-2])

plt.figure()
plt.title('Spot prices in Oslo')
data_daily.Oslo.plot()
data_daily.Oslo_predicted.plot(color='red')
plt.legend()

###############################################################################
###############################################################################
###############################################################################
# Task 2
# In the next step to improve the forecasts by adding historical hydro level
# into the model and forecasting the hourly power prices.

# I did not have time to tackle this anymore
# A different model e.g. VAR needs to be used which allows for more than one
# variable

###############################################################################
# Outlook
# The model can be improved by several steps
# 1) Treat summer/winter time correctly
# 2) Check/filter special days as e.g. 17.May in Norway
# 3) Feature engineering: Weekday, Autumn/Spring tag, group regions
# 4) Test other models and apply a grid search for optimal hyperparameters
# 5) At this point, I would probably rethink my whole approach. As there seemed
# to be only vague trends in the data, I would maybe switch to an entirely
# feature based approach (or a combination if it exists!?), e.g. random forest.
# Using weekday, season and average price of last month as main features.
# 
# I didn't have time to reach task 2. Conceptually, it could be imporved by
# including production prognosis from wind and sun (weather forcaste?)
# Also the power price in neighboring countries could be considered although
# there is a strong cross-correlation (prices are obviously not independent)
# and overfitting could happen
# overfitting