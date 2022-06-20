from  scipy.stats import skew, kurtosis, shapiro
import seaborn as sns

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.dates as mdates
import scipy.stats
import pylab
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import sys
import warnings
import pickle

class arimaModel:
    def __init__(self, datapath, category_name):
        warnings.filterwarnings("always", category=UserWarning, module='arimaModel')
        # This will be used to search for the reading files in the directory.
        self.datapath = datapath
        folderName = "Model_incercam9_" + category_name
        self.folderPath = self.create(folderName)
        self.modelName = "arima_"+category_name+'.pkl'
        self.df = pd.read_csv(datapath)
        
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time')
        self.df = self.df.set_index('time')
        
    def create(self, dirName):
        if not os.path.exists('../' + dirName):
            os.mkdir('../' + dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")
            
        return os.path.abspath('../' + dirName)

    # Data is standardized in order to allow application of models that are sensitive to scale, like neural networks or svm.
    # Remember that distribution shape is maintained, it only changes first and second momentum (mean and standard deviation)
    def normalize(self, df):
        ss_1 = StandardScaler()
        df_2 = pd.DataFrame(ss_1.fit_transform(df), index = df.index, columns = df.columns)
        return(df_2,ss_1)
    
    def norm_tra(self, df_1, ss_x):
        df_2 = pd.DataFrame(ss_x.transform(df_1), index = df_1.index, columns = df_1.columns)
        return(df_2)

    def split_data(self, s_df):

        df_train = s_df[s_df.index < "2018-04-01"]
        df_test = s_df[s_df.index >= "2018-04-01"]
        
        return df_train, df_test


    def fit_ARIMA(self, x_train, y_train):
        exogenous_features = set(x_train.columns.values) - set(['movave_3', 'movstd_3', 'movave_7', 'movstd_7', 'movave_30', 'movstd_30', 'q10', 'q50', 'q90'])
        exogenous_features = list(exogenous_features)
        print("exogenous features: ", exogenous_features)

        #model = auto_arima(y_train.avg, exogenous=x_train[exogenous_features], trace=True, error_action="ignore",
        #suppress_warnings=True, seasonal = True, m=48, start_p = 0, max_p = 2, max_q = 5, maxiter = 15, max_P=2, max_Q = 2, d = 0, D = 1)
        #model = ARIMA(endog = y_train.avg, exog = exogenous_features, order =[1,0,3], seasonal_order = [0,1,0,48])
        model= SARIMAX(y_train.avg,
                    freq = '30T',
                    exog=x_train[exogenous_features],
                    order=(0,0,0),
                    seasonal_order = (1,1,1,48),
                    enforce_invertibility=False, enforce_stationarity=False, simple_differencing = False)
        fit = model.fit(maxiter=50)
        return fit #model

        
    def predict(self, model, x_test, y_test, ss):
        exogenous_features = set(x_test.columns.values) - set(['movave_3', 'movstd_3', 'movave_7', 'movstd_7', 'movave_30', 'movstd_30', 'q10', 'q50', 'q90'])
        exogenous_features = list(exogenous_features)
        #forecast, confint = model.predict(len(x_test), exogenous = x_test[exogenous_features], return_conf_int = True)
        #new_model = model.append(endog = y_test.avg, exog = x_test[exogenous_features], refit = False)
        fcast = model.get_forecast(steps = len(x_test), index = y_test.index, freq = '30T', exog = x_test[exogenous_features])
        #fc, se, conf = model.forecast(len(x_test), exog = x_test[exogenous_features], alpha=0.05)  # 95% conf
        #res = pd.DataFrame(index = x_test.index)
        #res['Forecast_ARIMAX'] = forecast
        #res['forecast_normal'] = res['Forecast_ARIMAX'] #ss.inverse_transform(res['Forecast_ARIMAX'].values.reshape(-1, 1))
        #fc_series = pd.Series(fc, index=y_test.index)
        #lower_series = pd.Series(conf[:, 0], index=y_test.index)
        #upper_series = pd.Series(conf[:, 1], index=y_test.index)
        #cf= pd.DataFrame(confint, index = res.index)

        return fcast #res, cf

    def residual_analysis(self, df_valid):
        residuals=df_valid.avg-df_valid.forecast_normal

        self.dicky_fuller_test(residuals, 0.05)

        residuals.plot()
        plt.savefig(self.folderPath + '/residual_analysis.pdf')
        plt.close()
        plt.cla()
        plt.clf()
        
    def print_corr(self, s_df, col='avg'):
        correlations = s_df.corr(method='pearson')
        print(correlations[col].sort_values(ascending=False).to_string())

    def evaluation(self, df_valid, model):
        RMSE = 0
        MAE = 0
        MAPE = 0
        days = 0
        for i in range(48, len(df_valid), 48):
            RMSE_d = np.sqrt(mean_squared_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal))
            MAE_d = mean_absolute_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal)
            MAPE_d = mean_absolute_percentage_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal)
            if(i==48):
                print("RMSE for 01-04-2018:", RMSE_d)
                print("\nMAE for 01-04-2018:", MAE_d)
                print("\nMAPE for 01-04-2018:", MAPE_d)
            RMSE = RMSE + RMSE_d
            MAE = MAE + MAE_d
            MAPE = MAPE + MAPE_d
            days = days + 1
        
        print("RMSE for 30-06-2018:", RMSE_d)
        print("\nMAE for 30-06-2018:", MAE_d)
        print("\nMAPE for 30-06-2018:", MAPE_d)
        print("RMSE of Auto ARIMAX:", RMSE/days)
        print("\nMAE of Auto ARIMAX:", MAE/days)
        print("\nMAPE of Auto ARIMAX:", MAPE/days)


        model.plot_diagnostics(figsize=(15, 12))
        plt.savefig(self.folderPath + '/diagnostics.pdf')
        plt.close()
        plt.cla()
        plt.clf()

        self.print_corr(df_valid, 'forecast_normal')
        #correlations = df_valid.corr(method='pearson')
        #print(correlations['Forecast_ARIMAX'].sort_values(ascending=False).to_string())
        
    def dicky_fuller_test(self, x, cutoff = 0.05):
        result = adfuller(x)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        if result[1]>cutoff:
            print("Fail to reject the null hypothesis (H0), the data is non-stationary")
        else:
            print("Reject the null hypothesis (H0), the data is stationary.")
        
    def main(self):
        df_train, df_test = self.split_data(self.df)
        x_train = df_train.drop(columns = ['avg'])
        y_train = df_train['avg'].to_frame()
        y_train_norm, ss = self.normalize(y_train)
        
        x_test = df_test.drop(columns = ['avg'])
        y_test = df_test['avg'].to_frame()
        y_test_norm = self.norm_tra(y_test, ss)
        
        model = self.fit_ARIMA(x_train, y_train)
        results = self.predict(model, x_test, y_test, ss)
        #result['avg'] = y_test.avg
        print(self.modelName + ": ")
        print(model.summary())
        
        #cf_integral = pd.DataFrame(ss.inverse_transform(cf), index = cf.index, columns = cf.columns)
        #print('conf int: ', cf)
        #print('normal conf int', cf_integral)
        print(type(results))
        print('results: \n', results)

        cf = results.conf_int()
        
        #plt.figure(figsize=(12,5), dpi=100)
        #plt.plot(y_train.avg, label='training')
        #plt.plot(y_test.avg, label='actual')
        #plt.plot(results, label='forecast')
        #plt.fill_between(lcf.index, lcf, ucf, 
        #                color='k', alpha=.15)
        #plt.title('Forecast vs Actuals')
        #plt.legend(loc='upper left', fontsize=8)

        #results.index = df_test.index.copy()
        #results.set_index('time', inplace = True)

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.plot(y_test.avg)
        ax.plot(results.predicted_mean)
        #ax.fill_between(y_test_norm.index,
        #                cf[0],
        #                cf[1],color='grey',alpha=.3)
        ax.fill_between(results.predicted_mean.index, cf.iloc[:,0], cf.iloc[:,1], color='grey', alpha=0.3);
        plt.savefig(self.folderPath + '/resultsPlot_w_ConfInt.pdf')
        plt.close()
        plt.cla()
        plt.clf()
        
        '''
        fig3, ax3 = plt.subplots(1, 1, figsize=(15, 5))
        ax3.plot(y_test.avg)
        ax3.plot(results.forecast_normal)
        ax3.fill_between(y_test.index,
                        cf_integral[0],
                        cf_integral[1],color='grey',alpha=.3)
        plt.savefig(self.folderPath + '/resultsPlot_wConfInt_normal.pdf')
        plt.close()
        plt.cla()
        plt.clf()
        '''
        
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 5))
        ax2.plot(y_test.avg)
        ax2.plot(results.predicted_mean)
        plt.savefig(self.folderPath + '/resultsPlot_normal.pdf')
        plt.close()
        plt.cla()
        plt.clf()
        
        #results_test = pd.concat([df_test, results], axis=1)]
        df_test['forecast_normal'] = results.predicted_mean

        self.residual_analysis(df_test)
        self.evaluation(df_test, model)
        
        model.save(self.folderPath +'/'+ self.modelName)
        #with open(self.folderPath +'/'+ self.modelName, 'wb') as pkl:
        #    pickle.dump(model, pkl)
        
        
        

