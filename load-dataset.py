#Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

mylynx_df = pd.read_csv("LYNXdata.csv", header=0,
                     names=['year', 'trappings'],
                     index_col=0)

mylynxts = pd.Series(mylynx_df['trappings'].values,
                     index=pd.date_range('31/12/1821', periods=114, freq='A-DEC'))


cumsum_lynx = np.cumsum(mylynxts)
#One way to plot
# plt.figure(figsize=(12,8))
# plt.subplot(2,1,1)
# plt.plot(mylynxts)
# plt.title('Lynx Trappings in Canada 1821-1934')
#
# plt.subplot(2,1,2)
# plt.plot(cumsum_lynx)
# plt.title('Cumsum of Lynx')
#
# plt.tight_layout()

# plt.xlabel('Year of Trappings')
# plt.ylabel('Number of Lynx Trapped')
# plt.legend(['Lynx per year', 'Cumulative total'])

#Other way to plot
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
# mylynxts.plot(ax=ax1)
# cumsum_lynx.plot(ax=ax2)
# ax1.set_title("Lynx Trappings")
# ax2.set_title("Cumsum of Lynx")

#print(cumsum_lynx.head())

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


# stationarity_test(mylynxts)

# plt.plot(np.random.normal(1, 3, 300))

# stationarity_test(np.random.normal(1, 3, 300))

# mydata = (3, 5, 3, 65, 64, 64, 65, 643, 546, 546, 544)
# plt.plot(mydata)
# stationarity_test(mydata)



# Autocorrelation and partial autocorrelation in the Lynx dataset
# Two plots on one sheet
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12,8))
# plot_acf(mylynxts, lags = 20, ax = ax1)
# plot_pacf(mylynxts, lags = 20, ax = ax2);


# Simple moving average (rolling mean)
# Note: the rolling methods are applicable only on pandas Series
#                                            and DataFrame objects
def plot_rolling(timeseries, window):
    rol_mean = timeseries.rolling(window).mean()
    rol_std = timeseries.rolling(window).std()

    fig = plt.figure(figsize=(12, 8))
    og = plt.plot(timeseries, color="blue", label="Original")
    mean = plt.plot(rol_mean, color="red", label="Rolling Mean")
    std = plt.plot(rol_std, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean and Standard Deviation (window = " + str(window) + ")")
    plt.show()


# plot_rolling(mylynxts, 10)

# plot_rolling(mylynxts, 30)


# Simple rolling calculation with minimum number of periods for the window
def plot_rolling_min(timeseries, window):
    rol_mean = timeseries.rolling(window, min_periods=1).mean()
    rol_std = timeseries.rolling(window, min_periods=1).std()

    fig = plt.figure(figsize=(12, 8))
    og = plt.plot(timeseries, color="blue", label="Original")
    mean = plt.plot(rol_mean, color="red", label="Rolling Mean")
    std = plt.plot(rol_std, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean and Standard Deviation (window = " + str(window) + ")")
    plt.show()


# plot_rolling_min(mylynxts, 30)


# Exponentially weighted moving average
# Note: the ewm method is applicable on pandas Series and DataFrame objects only
def plot_ewma(timeseries, alpha):
    expw_ma = timeseries.ewm(alpha=alpha).mean()

    fig = plt.figure(figsize = (12, 8))
    og_line = plt.plot(timeseries, color = "blue", label = "Original")
    exwm_line = plt.plot(expw_ma, color = "red", label = "EWMA")
    plt.legend(loc = "best")
    plt.title("EWMA (alpha= "+str(alpha)+")")
    plt.show()


plot_ewma(mylynxts, 0.3)

print("Complete")