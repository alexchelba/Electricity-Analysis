import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import sys
import pickle
from utils.home_features_parse import featureParser
from utils.home_categories_parse import categoriesParser
from ARIMA.modelHelper import arimaModel
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import ObjectProperty
from kivy.uix.checkbox import CheckBox
import threading
import time
from kivy.clock import mainthread, Clock
from functools import partial
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


def processData(htype, nrppl, buildEra):
    bldEra = buildEra
    htp = htype
    if htype == 'flat':
        htp = htype
        if buildEra == 'Before 1900':
            bldEra = 'before 1900'
        elif buildEra == 'After 1965' or buildEra == 'After 1900 and before 1965':
            bldEra = 'after 1900'
    elif htype == 'house_or_bungalow':
        htp = "house"
        if buildEra == 'Before 1900' or buildEra == 'After 1900 and before 1965':
            bldEra = 'before 1965'
        elif buildEra == 'After 1965':
            bldEra = 'after 1965'

    name = htp + "_" + nrppl + "_" + bldEra
    #datapath = "generated_data2/" + name
    return name

def chooseARIMAModel(fileName):
    return os.path.normpath(os.path.join(os.getcwd(), "arima_nonseasonal/Model_" + fileName + "/arima_" + fileName + ".pkl"))

def chooseDataset(fileName):
    return os.path.normpath(os.path.join(os.getcwd(), "generated_data2/" + fileName + ".zip"))

def dicky_fuller_test(x, cutoff = 0.05):
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

def evaluation(df_valid):
    RMSE = 0
    MAE = 0
    MAPE = 0
    days = 0
    for i in range(48, len(df_valid), 48):
        RMSE_d = np.sqrt(mean_squared_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal))
        MAE_d = mean_absolute_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal)
        MAPE_d = mean_absolute_percentage_error(df_valid.iloc[(i-48):i].avg, df_valid.iloc[(i-48):i].forecast_normal)
        #if(i==48):
        #    print("RMSE for 01-04-2018:", RMSE_d)
        #    print("\nMAE for 01-04-2018:", MAE_d)
        #    print("\nMAPE for 01-04-2018:", MAPE_d)
        RMSE = RMSE + RMSE_d
        MAE = MAE + MAE_d
        MAPE = MAPE + MAPE_d
        days = days + 1
    
    #print("RMSE for 30-06-2018:", RMSE_d)
    #print("\nMAE for 30-06-2018:", MAE_d)
    #print("\nMAPE for 30-06-2018:", MAPE_d)
    print("Average daily RMSE:", RMSE/days)
    print("\nAverage daily MAE:", MAE/days)
    print("\nAverage daily MAPE:", MAPE/days)

def residual_analysis(df_valid):
    residuals=df_valid.avg-df_valid.forecast_normal
    residuals.plot()
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()

    dicky_fuller_test(residuals, 0.05)

def applyModel(model, fileName):
    if model == "ARIMA":
        m = chooseARIMAModel(fileName)
        db = chooseDataset(fileName)
        print(m)
        print('\n')
        print(db)
        print('\n')
        df = pd.read_csv(db)
        print(df.head())
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df = df.set_index('time')
        df = df[df.index >= '2018-04-01']
        with open(m, 'rb') as pkl:
            dff = df.drop(columns = ['avg'])
            #exogenous_features = set(dff.columns.values) - set(['movave_3', 'movstd_3', 'movave_7', 'movstd_7', 'movave_30', 'movstd_30', 'q10', 'q50', 'q90'])
            #exogenous_features = list(exogenous_features)
            #exogenous_features.sort()
            exogenous_features = ['weekend', 'qtr', 'month', 'season', 'hour', 'day', 'daylight', 'business hour']
            print("exog: ", exogenous_features)
            model = SARIMAXResults.load(pkl)
            pickle_preds = model.predict(n_periods=len(df), index = df.index, freq = '30T', exogenous = dff[exogenous_features], dynamic=False)
            preds = pd.Series(pickle_preds, index = df.index)
            df['forecast_normal'] = preds
            print("pickle preds: \n")
            print(type(preds))
            print(preds)

            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.plot(df.avg)
            ax.plot(df.forecast_normal)
            #ax.fill_between(y_test_norm.index,
            #                cf[0],
            #                cf[1],color='grey',alpha=.3)
            #ax.fill_between(results.index, results['mean_ci_lower'], results['mean_ci_upper'], color='grey', alpha=0.3);
            plt.show()
            plt.close()
            plt.cla()
            plt.clf()

            residual_analysis(df)
            evaluation(df)

    else:
        print("Model not recognised.")

def main(htype, nrppl, buildEra, model):
    print(htype)
    fileName = processData(htype, nrppl, buildEra)
    applyModel(model, fileName)
    print(fileName)

class MyGrid(Widget):
    flatType = ObjectProperty(None)
    houseType = ObjectProperty(None)
    arimaType = ObjectProperty(None)
    #gpType = ObjectProperty(None)
    htype = ""
    nrppl = ""
    buildEra = ""
    model = ""
    selectnrppl = ObjectProperty(None)
    selectbuildEra = ObjectProperty(None)
    stop = threading.Event()

    def checkbox_click(self, instance, value, hh):
        if value is True:
            self.htype = hh
        else:
            self.htype = ""

    def checkbox_click_m(self, instance, value, hh):
        if value is True:
            self.model = hh
        else:
            self.model = ""

    def assignNrPpl(self):
        self.nrppl = self.selectnrppl.text

    def assignBuildEra(self):
        self.buildEra = self.selectbuildEra.text

    def btn(self):
        print("htype:", self.htype, "nrppl:", self.nrppl,
                "build era:", self.buildEra, "model:", self.model)
        if self.htype == "" or self.nrppl == "" or self.buildEra == "" or self.model == "":
            print("All fields are required.")
        else:
            #mythread = threading.Thread(target = main, args = (self.htype, self.nrppl, self.buildEra, self.model))
            main(self.htype, self.nrppl, self.buildEra, self.model)
            #mythread.start()
            #mythread.join()
            self.reset_form()
            #main(self.htype, self.nrppl, self.buildEra, self.model)


    #@mainthread
    def reset_form(self):
        self.htype = ""
        self.nrppl = ""
        self.buildEra = ""
        self.model = ""
        self.flatType.active = False
        self.houseType.active = False
        self.arimaType.active = False
        #self.gpType.active = False
        self.selectnrppl.text = "Pick number of people"
        self.selectbuildEra.text = "Pick the era"

class FormApp(App):
    def build(self):
        return MyGrid()

if __name__ == "__main__":
	FormApp().run()