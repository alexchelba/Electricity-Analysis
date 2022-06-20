from  scipy.stats import skew, kurtosis, shapiro
import seaborn as sns

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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

class naiveModel:
    def __init__(self, datapath, category_name):
        warnings.filterwarnings("always", category=UserWarning, module='naiveModel')
        # This will be used to search for the reading files in the directory.
        self.datapath = datapath
        folderName = "naive_" + category_name
        self.folderPath = self.create(folderName)
        self.modelName = "naive_"+category_name+'.pkl'
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

    def split_data(self, s_df):

        df_train = s_df[s_df.index < "2018-04-01"]
        df_test = s_df[s_df.index >= "2018-04-01"]
        
        return df_train, df_test
        
    def predict(self, x_test):
        res = x_test.avg.shift(7*48)
        res = res['2018-04-01':]
        #res['forecast_normal'] = x_test.avg.shift(7*48)

        return res

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

    def evaluation(self, df_valid):
        RMSE = 0
        MAE = 0
        MAPE = 0
        days = 0
        for i in range(48, len(df_valid), 48):
            RMSE_d = np.sqrt(mean_squared_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal))
            MAE_d = mean_absolute_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal)
            MAPE_d = mean_absolute_percentage_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal)
            if(i==48):
                print("RMSE for 01-04-2018 naive:", RMSE_d)
                print("\nMAE for 01-04-2018 naive:", MAE_d)
                print("\nMAPE for 01-04-2018 naive:", MAPE_d)
            RMSE = RMSE + RMSE_d
            MAE = MAE + MAE_d
            MAPE = MAPE + MAPE_d
            days = days + 1
        
        print("RMSE naive for 30-06-2018:", RMSE_d)
        print("\nMAE naive for 30-06-2018:", MAE_d)
        print("\nMAPE naive for 30-06-2018:", MAPE_d)
        print("RMSE of naive:", RMSE/days)
        print("\nMAE of naive:", MAE/days)
        print("\nMAPE of naive:", MAPE/days)


        #model.plot_diagnostics(figsize=(15, 12))
        #plt.savefig(self.folderPath + '/diagnostics.pdf')
        #plt.close()
        #plt.cla()
        #plt.clf()

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
        
        x_test = df_test.drop(columns = ['avg'])
        y_test = df_test['avg'].to_frame()
        
        results = self.predict(self.df)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 5))
        ax2.plot(y_test.avg)
        ax2.plot(results)
        plt.savefig(self.folderPath + '/resultsPlot_normal.pdf')
        plt.close()
        plt.cla()
        plt.clf()
        
        #results_test = pd.concat([df_test, results], axis=1)
        df_test['forecast_normal'] = results
        self.residual_analysis(df_test)
        self.evaluation(df_test)