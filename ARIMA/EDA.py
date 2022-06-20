from  scipy.stats import skew, kurtosis, shapiro
import seaborn as sns

import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.dates as mdates
import scipy.stats
import pylab
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import sys
import warnings

class arimaModelHelper:
    def __init__(self, datapath, category_name):
        warnings.filterwarnings("always", category=UserWarning, module='arimaModel')
        # This will be used to search for the reading files in the directory.
        self.datapath = datapath
        folderName = "EDA_" + category_name
        self.folderPath = self.create(folderName)
        self.archiveName = category_name
        self.df = pd.read_csv(datapath)
        
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time')
        self.df = self.df.set_index('time')
        #self.df = self.df['2017-06-14 07:30':]
        
    def create(self, dirName):
        if not os.path.exists('../' + dirName):
            os.mkdir('../' + dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists")
            
        return os.path.abspath('../' + dirName)
        
    def edit_dataset(self, df):
        
        #df.loc[df.number<10, ['avg']] = np.nan
        start_idx = df.avg.first_valid_index()
        end_idx = df.avg.last_valid_index()
        if start_idx is not None:
            df = df[start_idx:]
        if end_idx is not None:
            df = df[:end_idx]
        df.loc[df.avg>1800, ['avg']] = np.nan
        df.avg = df.avg.fillna(value = df.groupby([df.index.year, 'month']).avg.transform('mean'))
        df.drop(columns=['max_val', 'number'], inplace = True)
        return df
            
    def moving_average(self, s_df):
        
        plt.rc('font', size=30)          # controls default text sizes

        s_df[["movave_3", "movstd_3"]] = s_df.avg.rolling(48*3).agg([np.mean, np.std])
        s_df[["movave_7", "movstd_7"]] = s_df.avg.rolling(48*7).agg([np.mean, np.std])
        s_df[["movave_30", "movstd_30"]] = s_df.avg.rolling(48*30).agg([np.mean, np.std])
        #s_df[["movave_90", "movstd_90"]] = s_df.avg.rolling(48*90).agg([np.mean, np.std])
        #s_df[["movave_365", "movstd_365"]] = s_df.avg.rolling(365).agg([np.mean, np.std])

        plt.figure(figsize=(20,16))
        plt.plot(s_df.index, s_df.avg, label = "Average Consumption")
        plt.plot(s_df.index, s_df.movave_7, label = "Weekly Moving Average")
        xticks = pd.date_range(min(s_df.index), max(s_df.index), periods = 5)
        plt.xticks([x.strftime('%Y-%m') for x in xticks])
        #s_df[["avg", "movave_7"]].plot(title="Weekly Energy Demand (Watts)")
        plt.title("Weekly Energy Demand")
        plt.ylabel("(Watts)")
        plt.legend()
        plt.savefig(self.folderPath + "/weeklyDemand.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

    def kurtosis_skewness(self, s_df): 
        # flat_2 or less people_After 1900

        # kurtosis = 11.98 > 3 => leptokurtic. having a wider or flatter
        # shape with fatter tails resulting in a greater chance of extreme positive or negative events.

        # skewness = 2.11 > 0 => data distribution is not symmetric and has a right tail.
        # Means the value of mean is the greatest one followed by median and then by mode

        mean = np.mean(s_df.avg.values)
        std = np.std(s_df.avg.values)
        skew0 = skew(s_df.avg.values)
        ex_kurt = kurtosis(s_df.avg, fisher = False)
        print("Skewness: {} \nKurtosis: {}".format(skew0, ex_kurt))
        return skew0, ex_kurt
    
    def plot_target_analysis(self, s_df):
        mean = s_df.avg.mean()
        std = s_df.avg.std()
        sns.distplot(s_df.avg)
        plt.title("Target Analysis")
        #plt.xticks(rotation=45)
        plt.axvline(x=mean, color='r', linestyle='-', label="mean: {0:.2f} Watts".format(mean))
        plt.axvline(x=mean+2*std, color='orange', linestyle='-', label = "+2 standard deviations")
        plt.axvline(x=max(0, mean-2*std), color='orange', linestyle='-', label = "-2 standard deviations")
        plt.legend()
        plt.xlabel("Watts")
        plt.savefig(self.folderPath + "/target_analysis.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

    def plot_quantiles(self, s_df): 
        data_rolling = s_df.avg.rolling(window=48*7)
        s_df['q10'] = data_rolling.quantile(0.1).to_frame("q10")
        s_df['q50'] = data_rolling.quantile(0.5).to_frame("q50")
        s_df['q90'] = data_rolling.quantile(0.9).to_frame("q90")
        fig, ax = plt.subplots()
        s_df[["q10", "q50", "q90"]].plot(title="Volatility Analysis: rolling percentiles", ax=ax)
        ax.legend(['10th percentile', '50th percentile', '90th percentile'])
        plt.ylabel("Watts")
        plt.savefig(self.folderPath + '/rolling_percentiles.pdf')
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

    def plot_CVs(self, s_df):
        s_df.groupby("qtr")["avg"].std().divide(s_df.groupby("qtr")["avg"].mean()).plot(kind="bar")
        plt.title("Coefficient of Variation (CV) by quarter")
        plt.savefig(self.folderPath + "/CV_Quarter.pdf")
        #plt.show()
        plt.close()
        plt.cla()
        plt.clf()

        s_df.groupby("month")["avg"].std().divide(s_df.groupby("month")["avg"].mean()).plot(kind="bar")
        plt.title("Coefficient of Variation (CV) by month")
        plt.savefig(self.folderPath + "/CV_Month.pdf")
        #plt.show()
        plt.close()
        plt.cla()
        plt.clf()

        s_df.groupby("season")["avg"].std().divide(s_df.groupby("season")["avg"].mean()).plot(kind="bar")
        plt.title("Coefficient of Variation (CV) by season")
        plt.savefig(self.folderPath + "/CV_Season.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

    def heteroscedasticity_week_month(self, s_df):
        s_df[["movstd_7", "movstd_30"]].plot(title="Heteroscedasticity analysis")
        plt.legend(['Weekly Variation', 'Monthly Variation'])
        plt.ylabel("Watts")
        plt.savefig(self.folderPath + "/heteroscedasticity_week_month.pdf")
        #plt.show()

    def seasonal_week_month(self, s_df):
        s_df[["movave_7", "movave_30"]].plot(title="Seasonal Analysis: Moving Averages")
        plt.legend(['Weekly MA', 'Monthly MA'])
        plt.ylabel("Watts")
        plt.savefig(self.folderPath + "/MA_Week_Month.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()
    
    def seasonality_distributions(self, s_df):
        sns.boxplot(data=s_df, x="qtr", y="avg")
        plt.title("Seasonality analysis: Distribution over quarters")
        plt.ylabel("Watts")
        plt.savefig(self.folderPath + "/Distr_Quarters.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

        sns.boxplot(data=s_df, x="weekend", y="avg")
        plt.title("Seasonality analysis: Distribution over weekdays compared to weekend")
        plt.ylabel("Watts")
        plt.savefig(self.folderPath + "/Distr_WeekDay_End.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()
        
        sns.boxplot(data=s_df, x="day", y="avg")
        plt.title("Seasonality analysis: Distribution over weekdays")
        plt.ylabel("Watts")
        plt.savefig(self.folderPath + "/Distr_Days.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

    def trend_analysis(self, s_df):
        sns.boxplot(data=s_df, x="month", y="avg")
        plt.title("Seasonality Analysis: Monthly Box-plot Distribution")
        plt.ylabel("(Watts)")
        plt.savefig(self.folderPath + "/Distr_Months.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

        sns.boxplot(data=s_df, x="season", y="avg")
        plt.title("Seasonality Analysis: Season Box-plot Distribution")
        plt.ylabel("(Watts)")
        plt.savefig(self.folderPath + "/Distr_Seasons.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        #plt.show()

        s_df['ix'] = range(0, len(s_df))
        sns.regplot(data=s_df[:100],x="ix", y="avg")
        plt.title("Trend analysis: Regression")
        plt.ylabel("(Watts)")
        plt.xlabel("Point index")
        plt.savefig(self.folderPath + "/TrendAnalysis.pdf")
        plt.close()
        plt.cla()
        plt.clf()
        s_df.drop(columns = ['ix'], inplace = True)
        #plt.show()

    # Data is standardized in order to allow application of models that are sensitive to scale, like neural networks or svm.
    # Remember that distribution shape is maintained, it only changes first and second momentum (mean and standard deviation)


#sns.kdeplot(s_df['avg'],shade=True)

#fig = go.Figure([go.Scatter(x=s_df.index, y=s_df['avg'])])
#fig.update_layout(
#    autosize=False,
#    width=1000,
#    height=500,
#    template='simple_white',
#    title='Consumption over time'
#)
#fig.update_xaxes(title="Date")
#fig.update_yaxes(title="Average Consumption")
#fig.show()
    
    def prob_plot(self, s_df):
        scipy.stats.probplot(s_df.avg, plot=pylab)
        #pylab.show()
        pylab.savefig(self.folderPath + "/probPlot.pdf")
        plt.close()
        plt.cla()
        plt.clf()

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

#dicky_fuller_test(s_df.avg)

#plt.rcParams.update({'figure.figsize': (10,10)})
#y = s_df['avg'].to_frame()


    def seasonal_decomposition(self, s_df):
        y = s_df['avg'].to_frame()
        # Multiplicative Decomposition 
        result_mul = seasonal_decompose(y, model='multiplicative',period = 48)

        # Additive Decomposition
        result_add = seasonal_decompose(y, model='additive',period = 48)

        # Plot
        plt.rcParams.update({'figure.figsize': (10,10)})
        result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
        result_add.plot().suptitle('Additive Decompose', fontsize=22)
        #plt.show()
        plt.savefig(self.folderPath + "/seasonalDecomposition.pdf")
        plt.close()
        plt.cla()
        plt.clf()

    def plot_acf_pacf(self, s_df):
        sm.graphics.tsa.plot_acf(s_df['avg'], lags=50,title='auto correlation of electricity consumption',zero=False)
        #plt.show()
        plt.xlabel("# of lags")
        plt.savefig(self.folderPath + "/acf_Plot.pdf")
        plt.close()
        plt.cla()
        plt.clf()

        sm.graphics.tsa.plot_pacf(s_df['avg'], lags=50,title='partial auto correlation of electricity consumption',zero=False)
        #plt.show()
        plt.xlabel("# of lags")
        plt.savefig(self.folderPath + "/pacf_Plot.pdf")
        plt.close()
        plt.cla()
        plt.clf()

    def print_corr(self, s_df, col='avg'):
        correlations = s_df.corr(method='pearson')
        print(correlations[col].sort_values(ascending=False).to_string())

    def saveFile(self):
        filename = self.archiveName
        compression_opts = dict(method='zip', archive_name=filename+'.csv')
        dirr = '../generated_data2/' + filename + '.zip'
        self.df.to_csv(dirr, compression=compression_opts)
        path = os.path.normpath(dirr)

    def main(self):
        #self.df = self.edit_dataset(self.df)
        #self.df.drop(columns = ['meantemp'], inplace = True)
        self.moving_average(self.df)
        #self.plot_target_analysis(self.df)
        #self.plot_quantiles(self.df)
        #self.plot_CVs(self.df)
        #self.heteroscedasticity_week_month(self.df)
        #self.seasonal_week_month(self.df)
        #self.seasonality_distributions(self.df)
        #self.trend_analysis(self.df)
        #self.prob_plot(self.df)
        #self.seasonal_decomposition(self.df)
        #self.plot_acf_pacf(self.df)
        #self.print_corr(self.df)
        #self.kurtosis_skewness(self.df)
        #self.dicky_fuller_test(self.df.avg)
        #self.saveFile()
