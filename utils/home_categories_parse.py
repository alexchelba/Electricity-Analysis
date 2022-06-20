import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import sys
import warnings
sys.path.insert(1, '../')
from API.IdealMetadataInterface import IdealMetadataInterface

class categoriesParser:
    def __init__(self, htype = None, nrppl = None, buildEra = None):

        # Make sure the warning is issued every time the user instantiates the class
        warnings.filterwarnings("always", category=UserWarning,
                                module='categoriesParser')

        # This will be used to search for the reading files in the directory.
        self.htype = htype
        self.nrppl = nrppl
        self.buildEra = buildEra
        self.init_dfs()
    
    def init_dfs(self):
        meta_folder_path = 'D:\\MInf_data\\metadata'
        self.meta_ideal = IdealMetadataInterface(meta_folder_path)
        self.homes = self.meta_ideal._metafile('home')['home']
        self.rooms = self.meta_ideal._metafile('room')['room']
        self.appliances = self.meta_ideal._metafile('appliance')['appliance']
        self.sensorboxes = self.meta_ideal._metafile('sensorbox')['sensorbox']
        self.sensors = self.meta_ideal._metafile('sensor')['sensor']
        self.people = self.meta_ideal._metafile('person')['person']
        self.locations = self.meta_ideal._metafile('location')['location']
        self.weatherfeeds = self.meta_ideal._metafile('weatherfeed')['weatherfeed']
    
    def getList(self):
        df = self.homes.copy()
        if self.htype is not None:
            df = df[df.hometype == self.htype]
        if self.buildEra is not None:
            if self.htype == 'flat':
                if self.buildEra == 'Before 1900':
                    df = df[(df.build_era=='Before 1850') | (df.build_era=='1850-1899')]
                elif self.buildEra == 'After 1900 and before 1965' or self.buildEra == 'After 1965':
                    df = df[(df.build_era=='2002 or later') | (df.build_era=='1965-1980')
                            | (df.build_era=='1981-1990') | (df.build_era=='1991-1995')
                            | (df.build_era=='1996-2001') | (df.build_era=='1900-1918')
                            | (df.build_era=='1919-1930') | (df.build_era=='1931-1944') | (df.build_era=='1945-1964')]
            elif self.htype == 'house_or_bungalow':
                if self.buildEra == 'After 1965':
                    df = df[(df.build_era=='2002 or later') | (df.build_era=='1965-1980')
                            | (df.build_era=='1981-1990') | (df.build_era=='1991-1995')
                            | (df.build_era=='1996-2001')]
                elif self.buildEra == 'After 1900 and before 1965' or self.buildEra == 'Before 1900':
                    df = df[(df.build_era=='Before 1850') | (df.build_era=='1850-1899')
                            | (df.build_era=='1900-1918') | (df.build_era=='1919-1930')
                            | (df.build_era=='1931-1944') | (df.build_era=='1945-1964')]
        if self.nrppl is not None:
            if self.nrppl == '2 or less people':
                df = df[(df.residents==1) | (df.residents==2)]
            elif self.nrppl == '3 or more people':
                df = df[df.residents>=3]
        lst = []
        for home in df.homeid:
            lst.append(home)
        return lst
