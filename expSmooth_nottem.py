# Required packages
import pandas as pd
import matplotlib.pyplot as plt
# Month plot
from statsmodels.graphics.tsaplots import month_plot
import pmdarima as pm
# Seasonal Decomposition
# Simple seasonal decomposition with statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose
# Decomposition based on stl - Package: stldecompose
# Install the library via PIP
from stldecompose import decompose
# Creating a forecast based on STL
from stldecompose import forecast
from stldecompose.forecast_funcs import (naive,
                                         drift,
                                         mean,
                                         seasonal_naive)
# Exponential smoothing function
from statsmodels.tsa.holtwinters import ExponentialSmoothing
help(ExponentialSmoothing)

nottem = pd.read_csv("nottem.csv", header = 0, names = ['Month', 'Temp'],
                     index_col = 0)
print(nottem.head())

# Conversion to a pandas Series object
nottemts = pd.Series((nottem.Temp).values,
                     index = pd.date_range('1920-01-31',
                                           periods = 240,
                                           freq = 'M'))

print(nottemts.head())

# Setting up the model Holt-Winters(A,N,A)
expsmodel = ExponentialSmoothing(nottemts, seasonal = "additive",
                                 seasonal_periods = 12)


# Fitting the model
# Default: optimized = True
# Optional: Insert smoothing coefficients
expsmodelfit = expsmodel.fit()

# Getting the alpha smoothing coefficient
print(expsmodelfit.params['smoothing_level'])


# Getting the gamma smoothing coefficient
print(expsmodelfit.params['smoothing_seasonal'])


# Prediction with the predict method
# Alternative: expsmodelfit.forecast(steps = 12)
expsfcast = expsmodelfit.predict(start = 240, end = 251)


# Plotting the predicted values and the original data
# plt.figure(figsize=(12,8))
# plt.plot(nottemts, label='data')
# plt.plot(expsfcast, label='HW forecast')
# # plt.xlim('1920-01-31T00:00:00.000000000','1941-12-31T00:00:00.000000000');
# # plt.ylim(30,70);
# plt.legend()


# Comparing the model and the original values
# How good is the model fit?
plt.figure(figsize=(12,8))
plt.plot(nottemts, label='data')
plt.plot(expsmodelfit.fittedvalues, label='HW model')
# plt.xlim('1920','1940'); plt.ylim(30,70);
plt.legend()

print("Complete..")