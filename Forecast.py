import pandas as pd
# pandas output display setup
pd.options.display.float_format = '{:,.2f}'.format
# menghilangkan warnings
import warnings
warnings.filterwarnings('ignore')
transaction = pd.read_csv('data_input/bank_trx.csv')
transaction.head()
# mengecek tipe data
transaction.dtypes
# Mengubah tipe data ke datetime64[ns]
transaction['transaction_date']=transaction['transaction_date'].astype('datetime64[ns]')
transaction.dtypes
# Proses yang sama dengan inplace
transaction= transaction.sort_values(by='transaction_date')
transaction.tail(10)
# membuat interval harian
transaction['date']=transaction['transaction_date'].dt.to_period('D')
transaction.head(3)
# menghitung jumlah trx per hari
daily_transaction= pd.crosstab(index = transaction['transaction_date'],
            columns=transaction['type'])
daily_transaction.head(6)
# Visualisasi Time Series
daily_transaction.plot(figsize=(15, 5), subplots=True);
# reset index daily transaction
daily_transaction = daily_transaction.reset_index()
daily_transaction.head()
daily_transaction = daily_transaction.rename_axis(columns=None)
daily_transaction
# rename column
daily_clean = daily_transaction.rename(columns= {'transaction_date': 'ds','Withdrawals':'y'})
daily_clean
# hapus kolom yang tidak diperlukan
daily_clean = daily_clean[['ds', 'y']]
daily_clean.head()
daily_clean.dtypes
daily_clean['ds'] = daily_clean['ds'].astype('datetime64[ns]')
daily_clean.dtypes
# load library
from prophet import Prophet
# mendefinisikan object model Prophet
model_prophet = Prophet()
model_prophet
# fitting model
# model.ft(df)
model_prophet.fit(daily_clean)
# membuat dataframe baru berisikan data waktu yang lama disertai data waktu baru
future= model_prophet.make_future_dataframe(periods=365,freq = 'B')
future
# Predict
forecast= model_prophet.predict(future)
forecast.tail()
forecast[['ds', 'trend', 'weekly', 'yearly', 'yhat']].tail()
model_prophet.plot(forecast);
model_prophet.plot_components(forecast);
model_prophet.plot(forecast);
from prophet.plot import add_changepoints_to_plot
fig = model_prophet.plot(forecast)
a = add_changepoints_to_plot(
    ax = fig.gca(), # plot
    m = model_prophet, # model
    fcst = forecast) # hasil forecast
# Membuat objek prophet tuning
model_trend = Prophet(changepoint_prior_scale= 1)
# fitting model
model_trend.fit(daily_clean)
# make time window to predict
future_trend = model_trend.make_future_dataframe(periods=365,
                                   freq = 'B')
# predict future dataframe
forecast_trend = model_trend.predict(future_trend)
# visualize components
model_trend.plot(forecast_trend);
# Visualisasi changepoint
fig = model_trend.plot(forecast_trend)
a = add_changepoints_to_plot(fig.gca(),
                             model_trend,
                             forecast_trend)
model_prophet.plot_components(forecast);
# tuning seasonality
model_seasonality = Prophet(yearly_seasonality=100)
# model fitting
model_seasonality.fit(daily_clean)
# forecasting
future_seasonality = model_seasonality.make_future_dataframe(periods=365, freq='B')
forecast_seasonality = model_seasonality.predict(future_seasonality)
# model tuned seasonality
model_seasonality.plot_components(forecast_seasonality);
# model tuned seasonality
model_seasonality.plot(forecast_seasonality);
# fitting model
model_monthly = Prophet()
# add seasonality monthly
model_monthly.add_seasonality(name = 'Bulanan',
                              period = 22, #weekdays
                              fourier_order = 5)
# model fitting
model_monthly.fit(daily_clean)
# forecasting
future_monthly = model_monthly.make_future_dataframe(periods=365, freq='B')
forecast_monthly = model_monthly.predict(future_monthly)
# visualize
model_monthly.plot_components(forecast_monthly);
model_monthly.plot(forecast_monthly);
# Additive Seasonality
model_monthly.plot(forecast_monthly);
# fitting model
model_multiplicative = Prophet(seasonality_mode= 'multiplicative')
# add seasonality
model_multiplicative.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_multiplicative.fit(daily_clean)
# forecasting
future_multiplicative = model_multiplicative.make_future_dataframe(periods=365, freq='B')
forecast_multiplicative = model_multiplicative.predict(future_multiplicative)
# visualize
model_multiplicative.plot(forecast_multiplicative);
model_multiplicative.plot_components(forecast_multiplicative);
forecast_multiplicative[['ds', 'trend', 'weekly', 'monthly', 'yearly', 'yhat']].tail()
# Add Holiday Effects
model_prophet.plot(forecast);
df_libur = pd.read_csv('data_input/holiday.csv', parse_dates=['ds'])
df_libur
# membuat objek model
model_holiday = Prophet(holidays=df_libur)
# fit
model_holiday.fit(daily_clean)
# Forecasting
future_holiday = model_holiday.make_future_dataframe(periods = 365, freq = 'B')
forecast_holiday = model_holiday.predict(future_holiday)
# visualize
model_holiday.plot(forecast_holiday);
model_holiday.plot_components(forecast_holiday);
model_multiplicative.plot(forecast_multiplicative)
daily_clean.describe()
transaction_train = daily_clean[daily_clean['ds']< '2018-01-01']
transaction_test = daily_clean[daily_clean['ds']>= '2018-01-01']
print(f'Train size: {transaction_train.shape}')
print(f'Test size: {transaction_test.shape}')
# fitting model
model_final = Prophet(holidays= df_libur, seasonality_mode= 'multiplicative')
# add seasonality
model_final.add_seasonality(name='monthly', period=30.5, fourier_order=5)
# fit dengan train
model_final.fit(transaction_train)
# forecasting
future_final = model_final.make_future_dataframe(periods=260 ,freq='B')
forecast_final = model_final.predict(future_final)
# visualize
import matplotlib.pyplot as plt
fig = model_final.plot(forecast_final)
plt.scatter(x = transaction_test['ds'], y = transaction_test['y'], color = 'red', s = 5);
plt.savefig('forecast_final.png') # Export plot to image (png)
# subset test data forecasts
forecast_test = forecast_final[forecast_final['ds']>= '2018-01-01']
forecast_test[['ds','yhat']].head(3)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true = transaction_test['y'],
                         y_pred= forecast_test['yhat'])
mae
# library preparation
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_true = transaction_test['y'],
                         y_pred= forecast_test['yhat'])
mape
# fitting model
model_tuning = Prophet(holidays=df_libur,
                      seasonality_mode = 'multiplicative',
                      changepoint_prior_scale=0.00305,
                      seasonality_prior_scale=3)
# add seasonality
model_tuning.add_seasonality(name='monthly' ,
                            period=30.5,
                            fourier_order=5)
# fit dengan train
model_tuning.fit(transaction_train)
# forecasting
future_tuning = model_tuning.make_future_dataframe(periods=260, freq="B")
forecast_tuning = model_tuning.predict(future_tuning)
forecast_test_tuning = forecast_tuning[forecast_tuning['ds'] >= '2018-01-01']
mae_tuning = mean_absolute_error(y_true=transaction_test['y'],
                                 y_pred=forecast_test_tuning['yhat'])
mae_tuning
mape_tuning = mean_absolute_percentage_error(y_true=transaction_test['y'],
                                            y_pred=forecast_test_tuning['yhat'])
mape_tuning
fig = model_tuning.plot(forecast_tuning)
plt.scatter(x = transaction_test['ds'], y = transaction_test['y'], color = 'red', s = 5);
