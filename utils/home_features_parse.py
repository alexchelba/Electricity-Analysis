import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import sys
import datetime
import warnings
sys.path.insert(1, '../')
from API.IdealDataInterface import IdealDataInterface
from API.IdealMetadataInterface import IdealMetadataInterface

pd.options.mode.chained_assignment = None  # default='warn'

class featureParser:
    def __init__(self, homeList, filename):

        # Make sure the warning is issued every time the user instantiates the class
        warnings.filterwarnings("always", category=UserWarning,
                                module='featureParser')

        # This will be used to search for the reading files in the directory.
        self.homeList = homeList
        
        if len(self.homeList) == 0:
            warnings.warn('Not enough data to make a prediction.')

        self.init_access()
        self.homeData = self._combine(homeList)
        self.filepath = self.saveFile(filename)
    
    def init_access(self):
        self.folder_path = '../data/sensordata'
        self.room_folder_path = '../data/sensordata'
        self.ideal = IdealDataInterface(self.folder_path)
        self.room_ideal = IdealDataInterface(self.room_folder_path)
        self.meta_folder_path = '../data/metadata'
        self.meta_ideal = IdealMetadataInterface(self.meta_folder_path)
    
    def mean(self, l):
        return sum(l) / float(len(l))
    
    def _combine(self, home_list):
        set_twilight_dict = {
            'January': datetime.time(hour=17, minute=0),
            'February': datetime.time(hour=18, minute=0),
            'March': datetime.time(hour=19, minute=0),
            'April': datetime.time(hour=21, minute=0),
            'May': datetime.time(hour=22, minute=0),
            'June': datetime.time(hour=23, minute=0),
            'July': datetime.time(hour=22, minute=45),
            'August': datetime.time(hour=21, minute=30),
            'September': datetime.time(hour=20, minute=0),
            'October': datetime.time(hour=18, minute=45),
            'November': datetime.time(hour=16, minute=50),
            'December': datetime.time(hour=16, minute=0)
        }
        sunrise_dict = {
            'January': datetime.time(hour=8, minute=30),
            'February': datetime.time(hour=7, minute=40),
            'March': datetime.time(hour=6, minute=30),
            'April': datetime.time(hour=6, minute=0),
            'May': datetime.time(hour=5, minute=0),
            'June': datetime.time(hour=4, minute=26),
            'July': datetime.time(hour=4, minute=45),
            'August': datetime.time(hour=5, minute=45),
            'September': datetime.time(hour=6, minute=45),
            'October': datetime.time(hour=7, minute=40),
            'November': datetime.time(hour=7, minute=50),
            'December': datetime.time(hour=8, minute=40)
        }
        print(home_list)

        '''
        weatherDF = pd.read_csv(self.folder_path + '/weatherreading.csv.gz')
        metaweatherDF = self.meta_ideal._metafile('weatherfeed')['weatherfeed']
        metahomes = self.meta_ideal._metafile('home')['home']
        '''
        home_data = self.ideal.get(homeid = home_list[0], category = 'electric-mains')
        
        '''
        loc = metahomes[metahomes.homeid==home_list[0]].get('location').item()
        feedid = metaweatherDF[(metaweatherDF.locationid == loc) & (metaweatherDF.weather_type=='temperature')].get('feedid').item()
        allWeather = weatherDF[weatherDF.feedid == feedid]
        allWeather['time'] = pd.to_datetime(allWeather['time']).dt.round('15min')
        allWeather = allWeather.astype({'value':int})
        allWeather.drop(columns = ['feedid'], inplace = True)
        allWeather = allWeather.set_index('time').resample('30T').mean()
        allWeather.reset_index(inplace = True)
        strn = 'value_' + str(0)
        allWeather.rename(columns={'value':strn}, inplace = True)
        '''

        ds = home_data[0]['readings']
        ds.interpolate(limit=300, inplace=True, limit_direction='both', limit_area='inside')
        df_resample = ds.resample('30T').mean()
        df_count = ds.resample('30T').count()
        df_resample.loc[df_count < 300] = np.nan
        df_all = df_resample.to_frame()
        name = 'electric-combined_'+str(0)
        df_all.rename(columns={'electric-combined':name}, inplace = True)
        df_all.reset_index(inplace = True)


        
        for idx,home_id in enumerate(home_list[1:], start=1):

            home_data = self.ideal.get(homeid = home_id, category = 'electric-mains')
            '''
            loc = metahomes[metahomes.homeid==home_id].get('location').item()
            feedid = metaweatherDF[(metaweatherDF.locationid == loc) & (metaweatherDF.weather_type=='temperature')].get('feedid').item()
            weatherInfo = weatherDF[weatherDF.feedid == feedid]
            weatherInfo['time'] = pd.to_datetime(weatherInfo['time']).dt.round('15min')
            weatherInfo = weatherInfo.astype({'value':int})
            weatherInfo.drop(columns = ['feedid'], inplace = True)
            weatherInfo = weatherInfo.set_index('time').resample('30T').mean()
            weatherInfo.reset_index(inplace = True)
            strn = 'value_' + str(idx)
            weatherInfo.rename(columns={'value':strn}, inplace = True)
            #weatherInfo.set_index('time')
            allWeather = pd.merge(allWeather, weatherInfo, how='outer', left_on = 'time', right_on = 'time')
            '''

            ds = home_data[0]['readings']
            ds.interpolate(limit=300, inplace=True, limit_direction='both', limit_area='inside')
            df_resample = ds.resample('30T').mean()
            df_count = ds.resample('30T').count()
            df_resample.loc[df_count < 300] = np.nan
            df = df_resample.to_frame()
            name = 'electric-combined_'+str(idx)
            df.rename(columns={'electric-combined':name}, inplace = True)
            df.reset_index(inplace = True)
            df_all = pd.merge(df_all, df, how='outer', left_on = 'time', right_on = 'time')
        
        df_all.sort_values('time')
        df_all.set_index('time', inplace = True)

        df = pd.DataFrame(index = df_all.index)


        '''
        df.reset_index(inplace=True)
        allWeather.set_index('time', inplace = True)
        meanWeather = pd.DataFrame(index = allWeather.index)
        meanWeather['meantemp'] = allWeather.mean(axis=1).round()
        meanWeather.reset_index(inplace = True)
        df = pd.merge(df, meanWeather, how='left', left_on = 'time', right_on = 'time')
        df.set_index('time', inplace = True)
        '''


        #df['avg'] = df_all.mean(axis=1)
        df['number'] = df_all.count(axis=1)
        df_all['avg'] = df_all.apply(lambda x: [y for y in x.values if ~np.isnan(y)][:10], axis=1)
        help_df = pd.DataFrame(df_all['avg'].values.tolist(), index = df.index)
        print("help_df: ")
        print(help_df.head())
        sd = help_df.mean(1)
        std = help_df.std(1)
        df['avg'] = sd
        df['max_val'] = df['avg'] + std
        df.loc[df['number']<5, ['avg']] = np.nan
        #df['avg'].interpolate(limit=1, inplace=True, limit_direction='both', limit_area='inside')
        #df['std'] = df_all.std(axis=1)
        #df['avg'].interpolate("linear", inplace=True)
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['monthname'] = df.index.month_name()
        #df['year'] = df.index.year
        df['sunday'] = df.index.weekday>5
        df['weekend'] = df.index.weekday>4
        df['business hour'] = (df.hour > 8) & (df.hour<19) & (~df.weekend) 
        df['weekend'] = df['weekend'].astype(int)
        df['sunday'] = df['sunday'].astype(int)
        df['business hour'] = df['business hour'].astype(int)
        df['weekend'] += df.sunday
        df['sunrise'] = df['monthname'].map(sunrise_dict)
        df['twilight'] = df['monthname'].map(set_twilight_dict)
        df['daysun'] = df.index.time >= df.sunrise
        df['daytwi'] = df.index.time <= df.twilight
        df['daylight'] = df['daysun'] & df['daytwi']
        df['daylight'] = df['daylight'].astype(int)

        season_dict = {
            1:4,
            2:4,
            3:1,
            4:1,
            5:1,
            6:2,
            7:2,
            8:2,
            9:3,
            10:3,
            11:3,
            12:4
        }
        df['season'] = df.month.map(season_dict)
        df["qtr"] = df.index.quarter
        df['day'] = df.index.weekday
        
        df.drop(columns = ['sunday', 'monthname', 'sunrise', 'twilight','daysun','daytwi'], inplace = True)
        start_idx = df.avg.first_valid_index()
        end_idx = df.avg.last_valid_index()
        if start_idx is not None:
            df = df[start_idx:]
        if end_idx is not None:
            df = df[:end_idx]
        #df['std'].interpolate("nearest", inplace=True)
        #df['min_val'] = np.where((df['avg'] - df['std']) < 0, 0,(df['avg'] - df['std']))
        #df['max_val'] = df['avg'] + df['std']
        return df
            
    def _parse(self, home_id):
        home_data = self.ideal.get(homeid = home_id, category = 'electric-mains')
        ds = home_data[0]['readings']
        df_resample = ds.resample('30T').sum()
        df_count = ds.resample('30T').count()
        df_resample.loc[df_count < 900] = np.nan
        df = df_resample.to_frame()
        return df

    def saveFile(self, filename):
        compression_opts = dict(method='zip', archive_name=filename+'.csv')
        dirr = '../regenerated/' + filename + '.zip'
        self.homeData.to_csv(dirr, compression=compression_opts)
        path = os.path.normpath(dirr)
        return path