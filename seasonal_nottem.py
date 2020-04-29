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

# Read in the 'nottem' dataset
# Make sure that nottem.csv is in the same folder as this python notebook
nottem = pd.read_csv("nottem.csv", header = 0, names = ['index', 'temp'],
                     index_col = 0)
print(nottem.head())

# Pandas DataFrame object with time stamp (monthly frequency)
nottem_df = pd.DataFrame((nottem.temp).values, columns = ['temperature'],
                     index = pd.date_range('1920-01-31',
                                           periods = 240,
                                           freq = 'M'))
print(nottem_df.head())


# Pandas Series object with time stamp (monthly frequency)
nottemts = pd.Series((nottem.temp).values,
                     index = pd.date_range('1920-01-31',
                                           periods = 240,
                                           freq = 'M'))

print(nottemts.head())

# Month_plot() requires the data to have a monthly (12 or 'M') frequency
# Alternative: quarter_plot() for dataset with a frequency of 4 or 'Q'
# fig, ax1 = plt.subplots(1, 1, figsize = (12,8))
# month_plot(nottemts, ax = ax1)
# plt.title("Month Plot of Nottem")
# plt.grid(axis = 'both')
# plt.tight_layout()


# Season plot
# Restructuring of nottem_df by pandas pivot_table
pivot_df = pd.pivot_table(nottem_df, index = nottem_df.index.month,
                          columns = nottem_df.index.year,
                          values = 'temperature')



# Add a new index to the pivot table
month_names = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
pivot_df.index = month_names

# Creating the season plot
# plt.figure(figsize=(12,8))
# plt.plot(pivot_df)
# plt.grid(axis = 'both')
# plt.legend(pivot_df.columns)
# plt.tight_layout()


# mySA = pm.auto_arima(nottemts, error_action="ignore", suppress_warnings = True,
#                      seasonal = True, m = 12, start_q = 1, start_p = 1,
#                      start_Q = 0, start_P = 0, max_order = 5, max_d = 1,
#                      max_D = 1, D = 1, stepwise = False, trace = True)
#
# print(mySA.summary())

# Additive or multiplicative decomposition
# plt.figure(figsize=(12,6))
# plt.plot(nottemts)


# By default model = "additive"
# For a multiplicative model use model = "multiplicative"
# nottem_decomposed = seasonal_decompose(nottemts)
#
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (14,9))
# nottemts.plot(ax = ax1)
# nottem_decomposed.trend.plot(ax = ax2)
# nottem_decomposed.seasonal.plot(ax = ax3)
# nottem_decomposed.resid.plot(ax = ax4)
# ax1.set_title("Nottem")
# ax2.set_title("Trend")
# ax3.set_title("Seasonality")
# ax4.set_title("Residuals")
# plt.tight_layout()


nottem_stl = decompose(nottemts, period=12)

# No NaN
print(nottem_stl.trend.head())

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (14,9))
# nottemts.plot(ax = ax1)
# nottem_stl.trend.plot(ax = ax2)
# nottem_stl.seasonal.plot(ax = ax3)
# nottem_stl.resid.plot(ax = ax4)
# ax1.set_title("Nottem")
# ax2.set_title("Trend")
# ax3.set_title("Seasonality")
# ax4.set_title("Residuals")
# plt.tight_layout()


# Eliminating the seasonal component
# nottem_adjusted = nottemts - nottem_stl.seasonal
# plt.figure(figsize=(12,8))
# nottem_adjusted.plot()


# Getting the seasonal component only
# Seasonality gives structure to the data
# plt.figure(figsize=(12,8))
# nottem_stl.seasonal.plot()

stl_fcast = forecast(nottem_stl, steps=12, fc_func=seasonal_naive,
                     seasonal = True)

stl_fcast.head()

# Plot of the forecast and the original data
plt.figure(figsize=(12,8))
plt.plot(nottemts, label='Nottem')
plt.plot(stl_fcast, label=stl_fcast.columns[0])
plt.legend()


input("Press enter to exit ;)")