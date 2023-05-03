import os
import ee
import datetime
import time
import sklearn
import importlib

import geopandas as gp
import pandas as pd
import numpy as np
import rsfuncs as rs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta
from sklearn import preprocessing

from tqdm import tqdm_notebook as tqdm

ee.Initialize()


# Load shapefile 
shp = gp.read_file("../shape/SGMA_Wyandotte Subbasin_2018.shp")

# Make EE objects from shapefiles 
area = rs.gdf_to_ee_poly(shp)

# Load RS data dict from rsfuncs.py
data = rs.load_data()

# Set start/end
strstart = '2016-01-01'
strend = '2020-12-31'

startdate = datetime.datetime.strptime(strstart, "%Y-%m-%d")
enddate = datetime.datetime.strptime(strend, "%Y-%m-%d")

# Aet
openet = rs.calc_monthly_sum(data['openet'], startdate, enddate, area)

aetdfs = {"openet":openet}


master_df = []
for i in [aetdfs]:
    for k,v in i.items():
        print(k,v.columns)
        newdf = v
        newdf.columns = [k]
        master_df.append(newdf)


finout = pd.concat(master_df, axis = 1)

finout.to_csv('../data/RS_analysis_dat_wyandotte_openet.csv')
