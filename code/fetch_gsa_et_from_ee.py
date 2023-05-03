#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:46:29 2023

@author: mmorphew
"""

import os
import ee
import datetime
import time
import sklearn
import importlib

import geopandas as gp
import pandas as pd
import numpy as np

from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta

import wxee
import xarray as xr
import rioxarray
from shapely.geometry import mapping


ee.Initialize()

make_dirs = False
base_output_dir = "../data/gsa_et/"
# Load GSAs shapefile and clean them so we only have "large GSAs"
gsa_shape = gp.read_file(
    "../shape/i03_Groundwater_Sustainability_Agencies_MapService.shp"
)
cv_shape = gp.read_file("../shape/Alluvial_Bnd.shp")
cv_shape.to_crs(3310, inplace=True)
gsa_shape.to_crs(3310, inplace=True)
cv_shape["geometry"] = cv_shape.geometry.buffer(10000)
gsa_cv = gsa_shape.within(cv_shape.loc[0, "geometry"])
gsa_cv = gsa_shape.loc[gsa_cv]
gsa_cv_cleaned_no_small = gsa_cv[gsa_cv["ShapeSTAre"] > 4e8]
gsa_names = []
gsa_ids = []

years = [
    "2001-01-01",
    "2002-01-01",
    "2003-01-01",
    "2004-01-01",
    "2005-01-01",
    "2006-01-01",
    "2007-01-01",
    "2008-01-01",
    "2009-01-01",
    "2010-01-01",
    "2011-01-01",
    "2012-01-01",
    "2013-01-01",
    "2014-01-01",
    "2015-01-01",
    "2016-01-01",
    "2017-01-01",
    "2018-01-01",
    "2019-01-01",
    "2020-01-01",
    "2021-01-01",
]
image_names = [
    "openet",
    "modis",
    # "gldas",
    # "nldas",
    "tc",
    # "gridmet",
    # "fldas"
]
image_locations = [
    ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("MODIS/006/MOD16A2"),
    # ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"),
    # ee.ImageCollection("NASA/NLDAS/FORA0125_H002"),
    ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE"),
    # ee.ImageCollection("IDAHO_EPSCOR/GRIDMET"),
    # ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"),
]
image_variables = [
    ["et_ensemble_mad", "et_ensemble_mad_min", "et_ensemble_mad_max"],
    ["ET", "PET"],
    # ["Evap_tavg", "PotEvap_tavg"],
    # ["potential_evaporation"],
    ["aet", "pet"],
    # ["etr", "eto"],
    # ["Evap_tavg"],
]
image_dict = {}

for i in range(len(years) - 1):
    for j in range(len(image_names)):
        if i < 15 and image_names[j] == "openet":
            continue
        image_dict[image_names[j] + "_" + years[i]] = (
            image_locations[j]
            .select(image_variables[j])
            .filterDate(years[i], years[i + 1])
        )
for index, row in gsa_cv_cleaned_no_small.iterrows():
    gsa_names.append(row["GSA_Name"])
    gsa_ids.append(row["GSA_ID"])

if make_dirs:
    for gsa_id in gsa_ids:
        os.makedirs(base_output_dir + "/" + str(gsa_id))

for i in range(len(gsa_names)):
    print("Processing GSA " + gsa_names[i])
    print("Number " + str(i) + " of " + str(len(gsa_names)) + " GSAs")
    for image_key in image_dict.keys():

        # Load shapefile
        print("Processing {}".format(image_key))
        if os.path.isfile(base_output_dir + str(gsa_ids[i]) + "/" + image_key + ".csv"):
            print("Already processed this dataset for this GSA, skipping")
            continue

        shp = gsa_cv_cleaned_no_small[gsa_cv_cleaned_no_small["GSA_ID"] == gsa_ids[i]]
        area = shp.area * 1e-6
        shp_degree = shp.to_crs(4326)

        image = image_dict[image_key]
        bounding_box = shp_degree["geometry"].bounds
        w_bounding_box = float(bounding_box["minx"]) - 0.01
        s_bounding_box = float(bounding_box["miny"]) - 0.01
        e_bounding_box = float(bounding_box["maxx"]) + 0.01
        n_bounding_box = float(bounding_box["maxy"]) + 0.01
        image_array = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box
            ),
            scale=1000,
        )
        image_array.rio.write_crs(4326, inplace=True)
        image_array = image_array.rio.set_spatial_dims(x_dim="x", y_dim="y")
        image_array_clipped = image_array.rio.clip(
            shp_degree.geometry.apply(mapping), shp_degree.crs, drop=True
        )

        image_array_clipped = image_array_clipped.where(image_array_clipped >= 0.0)
        image_array_clipped = image_array_clipped * 1e-6 * float(area)
        if "modis" in image_key:
            image_array_clipped = image_array_clipped * 0.1

        image_array_timeseries = image_array_clipped.mean(["x", "y"], skipna=True)
        print(image_array_timeseries.min())
        print(image_array_timeseries.max())
        image_array_df = image_array_timeseries.to_dataframe()
        image_array_df.to_csv(
            base_output_dir + str(gsa_ids[i]) + "/" + image_key + ".csv"
        )


# https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2
# data['modis_aet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "ET", 0.1, 25000]
# data['modis_pet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "PET", 0.1, 1000]

# https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H
# data['gldas_aet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'Evap_tavg', 86400*30 / 240 , 25000]   # kg/m2/s --> km3 / mon , noting 3 hrly images
# data['gldas_pet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'PotEvap_tavg', 1 / 240, 25000]

# https://developers.google.com/earth-engine/datasets/catalog/NASA_NLDAS_FORA0125_H002
# data['nldas_pet'] = [ee.ImageCollection('NASA/NLDAS/FORA0125_H002'), 'potential_evaporation', 1, 25000]

# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE
# data['tc_aet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "aet", 0.1 , 25000]
# data['tc_pet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "pet", 0.1, 25000]

# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
# data['gmet_etr'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "etr", 1 , 25000]
# data['gmet_eto'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "eto", 1, 25000]

# https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001
# data['fldas_aet'] = [ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001'), "Evap_tavg", 86400*30, 25000]
# https://developers.google.com/earth-engine/datasets/catalog/OpenET_ENSEMBLE_CONUS_GRIDMET_MONTHLY_v2_0
# data['openet'] = [ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0'), "et_ensemble_mad", 1, 25000]
