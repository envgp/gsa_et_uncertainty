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

output_dir = '../data/gsa_et/'
# Load GSAs shapefile and clean them so we only have "large GSAs"
gsa_shape = gp.read_file('../shape/i03_Groundwater_Sustainability_Agencies_MapService.shp')
cv_shape = gp.read_file('../shape/Alluvial_Bnd.shp')
cv_shape.to_crs(3310, inplace=True)
gsa_shape.to_crs(3310, inplace=True)
cv_shape['geometry'] = cv_shape.geometry.buffer(10000)
gsa_cv = gsa_shape.within(cv_shape.loc[0, 'geometry'])
gsa_cv = gsa_shape.loc[gsa_cv]
gsa_cv_cleaned_no_small = gsa_cv[gsa_cv['ShapeSTAre'] > 4e8]
gsa_names = []
gsa_ids = []

for index, row in gsa_cv_cleaned_no_small.iterrows():
    gsa_names.append(row['GSA_Name'])
    gsa_ids.append(row['GSA_ID'])

for i in range(len(gsa_names)):
 
    # Load shapefile 
    print("Processing {}".format(gsa_names[i]))
    if os.path.isfile(output_dir+'RS_analysis_dat_{}.csv'):
        print("Already processed this GSA, skipping")
        continue
        
    shp = gsa_cv_cleaned_no_small[gsa_cv_cleaned_no_small['GSA_ID']==gsa_ids[i]]
    #shp['geometry'] = shp['geometry'].simplify(10000000)

    # Make EE objects from shapefiles 
    area = rs.gdf_to_ee_poly(shp)

    # Load RS data dict from rsfuncs.py
    data = rs.load_data()

    # Set start/end
    strstart = '2001-01-01'
    strend = '2022-12-31'

    startdate = datetime.datetime.strptime(strstart, "%Y-%m-%d")
    enddate = datetime.datetime.strptime(strend, "%Y-%m-%d")

    # Aet
    modis_aet = rs.calc_monthly_sum(data['modis_aet'], startdate, enddate, area)
    gldas_aet = rs.calc_monthly_sum(data['gldas_aet'], startdate, enddate, area)
    tc_aet = rs.calc_monthly_sum(data['tc_aet'], startdate, enddate, area)
    fldas_aet = rs.calc_monthly_sum(data['fldas_aet'], startdate, enddate, area)
    openet = rs.calc_monthly_sum(data['openet'], "2016-01-01", enddate, area)

    # PET
    gldas_pet = rs.calc_monthly_sum(data['gldas_pet'], startdate, enddate, area)
    modis_pet = rs.calc_monthly_sum(data['modis_pet'], startdate, enddate, area)
    nldas_pet = rs.calc_monthly_sum(data['nldas_pet'], startdate, enddate, area)
    tc_pet = rs.calc_monthly_sum(data['tc_pet'], startdate, enddate, area)
    gmet_eto = rs.calc_monthly_sum(data['gmet_eto'], startdate, enddate, area)

    aetdfs = {"aet_modis":modis_aet, "aet_gldas":gldas_aet, "aet_tc":tc_aet, "aet_fldas":fldas_aet, "openet": openet}
    petdfs = {"pet_modis":modis_pet, "pet_gldas":gldas_pet, "pet_tc":tc_pet, "pet_nldas":nldas_pet, 'pet_gmet':gmet_eto }

    master_df = []
    for i in [aetdfs, petdfs, smdfs]:
        for k,v in i.items():
            print(k,v.columns)
            newdf = v
            newdf.columns = [k]
            master_df.append(newdf)

    finout = pd.concat(master_df, axis = 1)
    finout.to_csv(output_dir+'RS_analysis_dat_{}.csv'.format(GSA_ID))
    print("Completed: ", i+1, " out of ", len(gsa_ids), " GSAs")
