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

gsa_shape = gp.read_file(
    "../shape/i03_Groundwater_Sustainability_Plan_Areas_MapService.shp"
)
cv_shape = gp.read_file("../shape/Alluvial_Bnd.shp")
gsa_shape.to_crs(3310, inplace=True)
cv_shape.to_crs(3310, inplace=True)
gsa_shape.to_crs(3310, inplace=True)
cv_shape["geometry"] = cv_shape.geometry.buffer(10000)
gsa_cv = gsa_shape.within(cv_shape.loc[0, "geometry"])
gsa_cv = gsa_shape.loc[gsa_cv]
gsa_cv_cleaned_no_small = gsa_cv
# Butte - 98
# Vina - 86
# Wyandotte - 99
gsa_id = 99
shp = gsa_cv_cleaned_no_small[gsa_cv_cleaned_no_small["GSP_ID"] == gsa_id]
area = shp.area * 1e-6
shp_degree = shp.to_crs(4326)


# ds = xr.open_mfdataset("../data/et_images/DisALEXI/*.nc")
# ds = xr.open_mfdataset("../data/et_images/eeMETRIC/*.nc")
# ds = xr.open_mfdataset("../data/et_images/geeSEBAL/*.nc")
# ds = xr.open_mfdataset("../data/et_images/openetmax/*.nc")
# ds = xr.open_mfdataset("../data/et_images/openetmean/*.nc")
# ds = xr.open_mfdataset("../data/et_images/openetmin/*.nc")
# ds = xr.open_mfdataset("../data/et_images/PT-JPL/*.nc")
# ds = xr.open_mfdataset("../data/et_images/SIMS/*.nc")
# ds = xr.open_mfdataset("../data/et_images/SSEBop/*.nc")
# ds = xr.open_mfdataset("../data/et_images/et_images_butte/*.nc")
# ds.to_netcdf("../data/et_images/et_images_butte/butte_openet_images.nc")
# ds_timeseries = ds.mean(["x", "y"], skipna=True)
# print(image_array_timeseries.min())
# print(image_array_timeseries.max())
# print(len(image_array_timeseries))

# ds_df = ds_timeseries.to_dataframe()
# ds_df.to_csv("../data/et_images/et_images_wyandotte/wyandotte_openet_timeseries.csv")

# combine="nested",
# concat_dim="time",
# )
# image_array = image_array.interpolate_na(dim="x")
# image_array.rio.write_crs(4326, inplace=True)
# ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")
# ds_clipped = ds.rio.clip(shp_degree.geometry.apply(mapping), shp_degree.crs, drop=True)

# image_array_clipped = image_array_clipped.where(image_array_clipped >= 0.0)
# ds_clipped.to_netcdf("~/Desktop/et_images_wyandotte/SSEBop_Wyandotte.nc")


def try_earth_engine(
    image,
    w_bounding_box,
    s_bounding_box,
    e_bounding_box,
    n_bounding_box,
    new_distance_w,
    new_distance_s,
    new_distance_e,
    new_distance_n,
):

    if w_bounding_box <= -120.005 and e_bounding_box >= -119.995:
        image_array_w = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                w_bounding_box + new_distance_w,
                s_bounding_box + new_distance_s,
                -120.02,
                n_bounding_box + new_distance_n,
            ),
            scale=250,
        )
        image_array_e = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                -119.98,
                s_bounding_box + new_distance_s,
                e_bounding_box + new_distance_e,
                n_bounding_box + new_distance_n,
            ),
            scale=250,
        )
        image_array = xr.merge([image_array_w, image_array_e])
    # w_bounding_box = -121.00  # east boundary of nan line, -119.995
    # e_bounding_box = -120.01  # west boundary of nan line, -120.005
    # s_bounding_box = 36.5
    # n_bounding_box = 37.25
    else:
        image_array = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                w_bounding_box + new_distance_w,
                s_bounding_box + new_distance_s,
                w_bounding_box + new_distance_e,
                s_bounding_box + new_distance_n,
            ),
            scale=250,
        )
    return image_array


def try_earth_engine_2(
    image, w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box,
):

    if w_bounding_box <= -120.005 and e_bounding_box >= -119.995:
        image_array_w = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                w_bounding_box, s_bounding_box, -120.02, n_bounding_box
            ),
            scale=250,
        )
        image_array_e = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                -119.98, s_bounding_box, e_bounding_box, n_bounding_box,
            ),
            scale=250,
        )
        image_array = xr.merge([image_array_w, image_array_e])
    # w_bounding_box = -121.00  # east boundary of nan line, -119.995
    # e_bounding_box = -120.01  # west boundary of nan line, -120.005
    # s_bounding_box = 36.5
    # n_bounding_box = 37.25
    else:
        image_array = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                w_bounding_box, s_bounding_box, w_bounding_box, s_bounding_box,
            ),
            scale=250,
        )
    return image_array


ee.Initialize()

make_dirs = False
base_output_dir = "../data/et_images"
# Load GSAs shapefile and clean them so we only have "large GSAs"
gsa_shape = gp.read_file(
    "../shape/i03_Groundwater_Sustainability_Plan_Areas_MapService.shp"
)
cv_shape = gp.read_file("../shape/Alluvial_Bnd.shp")
cv_shape.to_crs(3310, inplace=True)
gsa_shape.to_crs(3310, inplace=True)
cv_shape["geometry"] = cv_shape.geometry.buffer(10000)

# years = [
#    "2016-01-01",
#    "2016-02-01",
#    "2017-01-01",
#    "2018-01-01",
#    "2019-01-01",
#    "2020-01-01",
#    "2021-01-01",
#    "2022-01-01",
# ]

years = [
    "20" + str(i) + "-" + str(j).zfill(2) + "-01"
    for i in range(16, 23)
    for j in range(1, 13)
]

image_names = [
    "openetmean",
    "openetmin",
    "openetmax",
    "geeSEBAL",
    "PT-JPL",
    "DisALEXI",
    "SIMS",
    "SSEBop",
    "eeMETRIC",
    # "modis",
    # "gldas",
    # "nldas",
    # "tc",
    # "gridmet",
    # "fldas"
]
image_locations = [
    ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/GEESEBAL/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/PTJPL/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/DISALEXI/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/SIMS/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/SSEBOP/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/EEMETRIC/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("MODIS/006/MOD16A2"),
    # ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"),
    # ee.ImageCollection("NASA/NLDAS/FORA0125_H002"),
    # ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE"),
    # ee.ImageCollection("IDAHO_EPSCOR/GRIDMET"),
    # ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"),
]

image_variables = [
    ["et_ensemble_mad"],
    ["et_ensemble_mad_min"],
    ["et_ensemble_mad_max"],
    ["et"],
    ["et"],
    ["et"],
    ["et"],
    ["et"],
    ["et"],
    # ["ET", "PET"],
    # ["Evap_tavg", "PotEvap_tavg"],
    # ["potential_evaporation"],
    # ["aet", "pet"],
    # ["etr", "eto"],
    # ["Evap_tavg"],
]
image_dict = {}

for i in range(len(years) - 1):
    for j in range(len(image_names)):
        image_dict[image_names[j] + "_" + years[i]] = (
            image_locations[j]
            .select(image_variables[j])
            .filterDate(years[i], years[i + 1])
        )
if make_dirs:
    for image_name in image_names:
        os.makedirs(base_output_dir + "/" + str(image_name))

for image_key in image_dict.keys():
    image = image_dict[image_key]
    # Load shapefile
    print("Processing {}".format(image_key))
    image_name = image_key.split("_")[0]
    if os.path.isfile(
        base_output_dir + "/" + str(image_name) + "/" + str(image_key) + ".nc"
    ):
        print("Already processed this item, skipping")
        continue

    shp = cv_shape
    shp_degree = shp.to_crs(4326)

    bounding_box = shp_degree["geometry"].bounds
    w_bounding_box = float(bounding_box["minx"]) - 2
    s_bounding_box = float(bounding_box["miny"]) - 2
    e_bounding_box = float(bounding_box["maxx"]) + 2
    n_bounding_box = float(bounding_box["maxy"]) + 2

    # middle_ns = (n_bounding_box + s_bounding_box) / 1000
    # middle_we = (w_bounding_box + e_bounding_box) /
    # distance_ns = abs(n_bounding_box - s_bounding_box) / 10
    # distance_we = abs(w_bounding_box - e_bounding_box) / 10
    # image_array.rio.write_crs(4326, inplace=True)

    # for j in range(1, 10):
    #    new_distance_s = distance_ns * (j - 1)
    #    new_distance_n = distance_ns * j
    #    for i in range(1, 10):
    #        new_distance_w = distance_we * (i - 1)
    #        new_distance_e = distance_we * i

    data_fetch_success = False
    while not data_fetch_success:
        try:
            image_array = try_earth_engine_2(
                image,
                w_bounding_box,
                s_bounding_box,
                e_bounding_box,
                n_bounding_box,
                # new_distance_w,
                # new_distance_s,
                # new_distance_e,
                # new_distance_n,
            )
            data_fetch_success = True
        except Exception:
            print("Google Earth Engine said no. Trying again...")
            continue
    # image_array = image_array.interpolate_na(dim="x")
    image_array.rio.write_crs(4326, inplace=True)
    # image_array = image_array.rio.set_spatial_dims(x_dim="x", y_dim="y")
    # image_array_clipped = image_array.rio.clip(
    #    shp_degree.geometry.apply(mapping), shp_degree.crs, drop=True
    # )

    # image_array_clipped = image_array_clipped.where(image_array_clipped >= 0.0)
    # image_array_clipped = image_array_clipped * 1e-6 * float(area)
    if "modis" in image_key:
        image_array = image_array * 0.1
        image_array = image_array.rename({"ET": "modis_ET", "PET": "modis_PET"})
    if "tc" in image_key:
        image_array = image_array.rename({"aet": "tc_ET", "pet": "tc_PET"})
    elif "geeSEBAL" in image_key:
        image_array = image_array.rename({"et": "geeSEBAL_ET"})
    elif "PT-JPL" in image_key:
        image_array = image_array.rename({"et": "PT-JPL_ET"})
    elif "DisALEXI" in image_key:
        image_array = image_array.rename({"et": "DisALEXI_ET"})
    elif "SIMS" in image_key:
        image_array = image_array.rename({"et": "SIMS_ET"})
    elif "SSEBop" in image_key:
        image_array = image_array.rename({"et": "SSEBop_ET"})
    elif "eeMETRIC" in image_key:
        image_array = image_array.rename({"et": "eeMETRIC_ET"})

    # image_array_timeseries = image_array_clipped.mean(["x", "y"], skipna=True)
    # print(image_array_timeseries.min())
    # print(image_array_timeseries.max())
    # print(len(image_array_timeseries))

    # image_array_df = image_array_timeseries.to_dataframe()
    # image_array_df.to_csv(
    #    base_output_dir + str(gsa_ids[i]) + "/" + image_key + ".csv"
    # )
    image_array.to_netcdf(
        base_output_dir + "/" + str(image_name) + "/" + str(image_key) + ".nc"
    )
    # print("saving image")
    # image_array.to_netcdf(base_output_dir + str(image_key) + ".nc")


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
# data['openet'] = [ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0'), "et_ensemble_mad", 1, 25000]
