import pandas as pd
import numpy as np
import statsmodels as sm
import matplotlib.pyplot as plt
# Getting the ARIMA modeling function
from statsmodels.tsa.arima_model import ARIMA
# ACF and PACF functions to test for autocorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Importing the Lynx dataset
# Make sure that LYNXdata.csv is in the same folder as this python notebook
mylynx_df = pd.read_csv("LYNXdata.csv", header = 0,
                        index_col = 0)

print(mylynx_df.head())

# Converting the DataFrame into a Series object
# ARIMA modeling requires a tuple index
mylynxts = pd.Series(mylynx_df['trappings'].values,
                 index = pd.DatetimeIndex(data = (tuple(pd.date_range('31/12/1821',
                                                                    periods = 114,
                                                                    freq = 'A-DEC'))),
                                            freq = 'A-DEC'))

print(mylynxts.head())

# Test for stationarity
def stationarity_test(timeseries):
    """"Augmented Dickey-Fuller test
    A test for stationarity"""
    from statsmodels.tsa.stattools import adfuller
    print("Results of Dickey-Fuller Test:")
    df_test = adfuller(timeseries, autolag = "AIC")
    df_output = pd.Series(df_test[0:4],
                          index = ["Test statistic", "p-value",
                                   "Number of lags used",
                                   "Number of observations used"])
    print(df_output)



stationarity_test(mylynxts)

# # ACF and PACF plots
# # Rule of thumb: Start with the plot that shows the least number of significant lags
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = plot_acf(mylynxts, lags=20, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = plot_pacf(mylynxts, lags=20, ax=ax2)

# ARIMA model setup
model = ARIMA(mylynxts, order=(4, 0, 0))
results_AR4 = model.fit()
# plt.figure(figsize=(12,8))
# plt.plot(mylynxts)
# plt.plot(results_AR2.fittedvalues, color='red')

print(results_AR4.resid.tail())

print((mylynxts - results_AR4.fittedvalues).tail())

print(np.mean(results_AR4.resid))


# Custom function to test for a normal distribution
def resid_histogram(data):
    import matplotlib.pyplot as plt
    from numpy import linspace
    from scipy.stats import norm

    plt.figure(figsize=(10,6))
    plt.hist(data, bins = 'auto', density = True, rwidth = 0.85,
             label = 'Residuals')
    mean_resid, std_resid = norm.fit(data)
    xmin, xmax = plt.xlim()
    curve_length = linspace(xmin, xmax, 100)
    bell_curve = norm.pdf(curve_length, mean_resid, std_resid)
    plt.plot(curve_length, bell_curve, 'm', linewidth = 2)
    plt.grid(axis='y', alpha = 0.2)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residuals vs Normal Distribution - Mean = '+str(round(mean_resid,2))+', Std = '+str(round(std_resid,2)))
    plt.show()


# resid_histogram(results_AR4.resid)

# ARIMA forecast
Fcast400 = results_AR4.predict(start = '31/12/1935',
                               end = '31/12/1945')

# Arima(2,0,2) model and forecast
model202 = ARIMA(mylynxts, order=(2, 0, 2))
results_M202 = model202.fit()
Fcast202 = results_M202.predict(start = '31/12/1935',
                                end = '31/12/1945')

# Forecast comparison
plt.figure(figsize = (12, 8))
plt.plot(mylynxts, linewidth = 2, label = "original")
plt.plot(Fcast400, color='red', linewidth = 2,
         label = "ARIMA 4 0 0")
plt.plot(Fcast202, color='blue', linewidth = 2,
         label = "ARIMA 2 0 2")
plt.legend()


print("Complete..")
