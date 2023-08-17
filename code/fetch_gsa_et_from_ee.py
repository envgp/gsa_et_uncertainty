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
import scipy


def try_earth_engine(
    image, w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box
):
    # if gsa_id == 42:  # 42 is a problem child
    #    image_array = image.wx.to_xarray(
    #        region=ee.Geometry.BBox(
    #            -119.98, s_bounding_box, e_bounding_box, n_bounding_box
    #        ),
    #        scale=250,
    #    )
    #    return image_array
    if w_bounding_box <= -120.005 and e_bounding_box >= -119.995:
        image_array_w = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                w_bounding_box, s_bounding_box, -120.02, n_bounding_box
            ),
            scale=250,
        )
        image_array_e = image.wx.to_xarray(
            region=ee.Geometry.BBox(
                -119.98, s_bounding_box, e_bounding_box, n_bounding_box
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
                w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box
            ),
            scale=250,
        )
    return image_array


ee.Initialize()

make_dirs = False
base_output_dir = "../data/gsa_et_gsp_take_3/"
# Load GSAs shapefile and clean them so we only have "large GSAs"
gsa_shape = gp.read_file(
    "../shape/i03_Groundwater_Sustainability_Plan_Areas_MapService.shp"
)
cv_shape = gp.read_file("../shape/Alluvial_Bnd.shp")
cv_shape.to_crs(3310, inplace=True)
gsa_shape.to_crs(3310, inplace=True)
cv_shape["geometry"] = cv_shape.geometry.buffer(10000)
gsa_cv = gsa_shape.within(cv_shape.loc[0, "geometry"])
gsa_cv = gsa_shape.loc[gsa_cv]
gsa_cv_cleaned_no_small = gsa_cv
gsa_ids = []
"""
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
    "2022-01-01",
]
"""
years = [
    "20" + str(i) + "-" + str(j).zfill(2) + "-01"
    for i in range(16, 23)
    for j in range(1, 13)
]

image_names = [
    # "openet",
    "openet_bitmask",
    # "geeSEBAL",
    # "PT-JPL",
    # "DisALEXI",
    # "SIMS",
    # "SSEBop",
    # "eeMETRIC",
    # "modis",
    # "gldas",
    # "nldas",
    # "tc",
    # "gridmet",
    # "fldas"
]
image_locations = [
    # ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"),
    ee.ImageCollection("OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("OpenET/GEESEBAL/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("OpenET/PTJPL/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("OpenET/DISALEXI/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("OpenET/SIMS/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("OpenET/SSEBOP/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("OpenET/EEMETRIC/CONUS/GRIDMET/MONTHLY/v2_0"),
    # ee.ImageCollection("MODIS/006/MOD16A2"),
    # ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"),
    # ee.ImageCollection("NASA/NLDAS/FORA0125_H002"),
    # ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE"),
    # ee.ImageCollection("IDAHO_EPSCOR/GRIDMET"),
    # ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"),
]
image_variables = [
    # ["et_ensemble_mad", "et_ensemble_mad_min", "et_ensemble_mad_max"],
    ["et_ensemble_mad_count", "et_ensemble_mad_index", "et_ensemble_sam"],
    # ["et"],
    # ["et"],
    # ["et"],
    # ["et"],
    # ["et"],
    # ["et"],
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
        # if i < 15 and np.isin(
        #    image_names[j],
        #    [
        #        "openet",
        #        "openet_bitmask",
        #        "geeSEBAL",
        #        "PT-JPL",
        #        "DisALEXI",
        #        "SIMS",
        #        "SSEBop",
        #        "eeMETRIC",
        #    ],
        # ):
        #    continue
        image_dict[image_names[j] + "_" + years[i]] = (
            image_locations[j]
            .select(image_variables[j])
            .filterDate(years[i], years[i + 1])
        )
for index, row in gsa_cv_cleaned_no_small.iterrows():
    gsa_ids.append(row["GSP_ID"])
print(gsa_ids)
if make_dirs:
    for gsa_id in gsa_ids:
        os.makedirs(base_output_dir + "/" + str(gsa_id))

# gsa_ids = [42]
"""
for i in range(len(gsa_ids)):
    print("Processing GSP with ID" + str(gsa_ids[i]))
    print("Number " + str(i) + " of " + str(len(gsa_ids)) + " GSAs")
    for image_key in image_dict.keys():

        # Load shapefile
        print("Processing {}".format(image_key))
        if os.path.isfile(base_output_dir + str(gsa_ids[i]) + "/" + image_key + ".csv"):
            print("Already processed this dataset for this GSA, skipping")
            continue

        shp = gsa_cv_cleaned_no_small[gsa_cv_cleaned_no_small["GSP_ID"] == gsa_ids[i]]
        area = shp.area * 1e-6
        shp_degree = shp.to_crs(4326)

        image = image_dict[image_key]
        shp_2 = cv_shape
        shp_2_degree = shp_2.to_crs(4326)
        bounding_box = shp_2_degree["geometry"].bounds
        w_bounding_box = float(bounding_box["minx"]) - 0.01
        s_bounding_box = float(bounding_box["miny"]) - 0.01
        e_bounding_box = float(bounding_box["maxx"]) + 0.01
        n_bounding_box = float(bounding_box["maxy"]) + 0.01

        data_fetch_success = False
        while not data_fetch_success:
            try:
                image_array = try_earth_engine(
                    image,
                    w_bounding_box,
                    s_bounding_box,
                    e_bounding_box,
                    n_bounding_box,
                    gsa_ids[i],
                )
                data_fetch_success = True
            except Exception:
                print("Google Earth Engine said no. Trying again...")
                continue
\
        if w_bounding_box <= -120.005 and e_bounding_box >= -119.995:
            image_array_w = image.wx.to_xarray(
                region=ee.Geometry.BBox(
                    w_bounding_box, s_bounding_box, -120.02, n_bounding_box
                ),
                scale=1000,
            )
            image_array_e = image.wx.to_xarray(
                region=ee.Geometry.BBox(
                    -119.98, s_bounding_box, e_bounding_box, n_bounding_box
                ),
                scale=1000,
            )
            image_array = xr.merge([image_array_w, image_array_e])
        # w_bounding_box = -121.00  # east boundary of nan line, -119.995
        # e_bounding_box = -120.01  # west boundary of nan line, -120.005
        # s_bounding_box = 36.5
        # n_bounding_box = 37.25
        else:
            image_array = image.wx.to_xarray(
                region=ee.Geometry.BBox(
                    w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box
                ),
                scale=1000,
            )

        image_array = image_array.interpolate_na(dim="x")
        image_array.rio.write_crs(4326, inplace=True)
        image_array = image_array.rio.set_spatial_dims(x_dim="x", y_dim="y")
        image_array_clipped = image_array.rio.clip(
            shp_degree.geometry.apply(mapping), shp_degree.crs, drop=True
        )

        image_array_clipped = image_array_clipped.where(image_array_clipped >= 0.0)
        image_array_clipped = image_array_clipped * 1e-6 * float(area)
        print(image_array_clipped)
        dropped_counts = []
        for i in range(12):
            array_values = (
                image_array_clipped["et_ensemble_mad_index"].isel(time=i).values
            )
            array_values = array_values[~np.isnan(array_values)]
            print(array_values)
            dropped_count = [0, 0, 0, 0, 0, 0]
            for value in array_values:
                binary_string = format(int(value), "#08b")[2:]
                print(binary_string)
                for i in range(len(binary_string)):
                    if binary_string[i] == "0":
                        dropped_count[i] += 1
            dropped_count.append(len(array_values))
            print(dropped_count)
            dropped_counts.append(dropped_count)

        dropped_counts = np.array(dropped_counts)

        if "modis" in image_key:
            image_array_clipped = image_array_clipped * 0.1
            image_array_clipped = image_array_clipped.rename(
                {"ET": "modis_ET", "PET": "modis_PET"}
            )
        if "tc" in image_key:
            image_array_clipped = image_array_clipped.rename(
                {"aet": "tc_ET", "pet": "tc_PET"}
            )
        elif "geeSEBAL" in image_key:
            image_array_clipped = image_array_clipped.rename({"et": "geeSEBAL_ET"})
        elif "PT-JPL" in image_key:
            image_array_clipped = image_array_clipped.rename({"et": "PT-JPL_ET"})
        elif "DisALEXI" in image_key:
            image_array_clipped = image_array_clipped.rename({"et": "DisALEXI_ET"})
        elif "SIMS" in image_key:
            image_array_clipped = image_array_clipped.rename({"et": "SIMS_ET"})
        elif "SSEBop" in image_key:
            image_array_clipped = image_array_clipped.rename({"et": "SSEBop_ET"})
        elif "eeMETRIC" in image_key:
            image_array_clipped = image_array_clipped.rename({"et": "eeMETRIC_ET"})

        image_array_timeseries = image_array_clipped.mean(["x", "y"], skipna=True)

        image_array_df = image_array_timeseries.to_dataframe()
        image_array_df["DisALEXI_dropped"] = dropped_counts[:, 0]
        image_array_df["EEMETRIC_dropped"] = dropped_counts[:, 1]
        image_array_df["GEESEBAL_dropped"] = dropped_counts[:, 2]
        image_array_df["PTJPL_dropped"] = dropped_counts[:, 3]
        image_array_df["SIMS_dropped"] = dropped_counts[:, 4]
        image_array_df["SSEBop_dropped"] = dropped_counts[:, 5]
        image_array_df["Total_pixels"] = dropped_counts[:, 6]

        image_array_df.to_csv(
            base_output_dir + str(gsa_ids[i]) + "/" + image_key + ".csv"
        )
"""


for image_key in image_dict.keys():

    # Load shapefile
    print("Processing {}".format(image_key))

    image = image_dict[image_key]
    shp_2 = cv_shape
    shp_2_degree = shp_2.to_crs(4326)
    bounding_box = shp_2_degree["geometry"].bounds
    w_bounding_box = float(bounding_box["minx"]) - 0.01
    s_bounding_box = float(bounding_box["miny"]) - 0.01
    e_bounding_box = float(bounding_box["maxx"]) + 0.01
    n_bounding_box = float(bounding_box["maxy"]) + 0.01

    data_fetch_success = False
    while not data_fetch_success:
        try:
            image_array = try_earth_engine(
                image, w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box,
            )
            data_fetch_success = True
        except Exception:
            print("Google Earth Engine said no. Trying again...")
            continue

    image_array = image_array.interpolate_na(dim="x")
    image_array.rio.write_crs(4326, inplace=True)
    image_array = image_array.rio.set_spatial_dims(x_dim="x", y_dim="y")
    for i in range(len(gsa_ids)):
        if os.path.isfile(base_output_dir + str(gsa_ids[i]) + "/" + image_key + ".csv"):
            print("Already processed this dataset for this GSA, skipping")
            continue

        shp = gsa_cv_cleaned_no_small[gsa_cv_cleaned_no_small["GSP_ID"] == gsa_ids[i]]
        area = shp.area * 1e-6
        shp_degree = shp.to_crs(4326)
        image_array_clipped = image_array.rio.clip(
            shp_degree.geometry.apply(mapping), shp_degree.crs, drop=True
        )

        image_array_clipped = image_array_clipped.where(image_array_clipped >= 0.0)
        print(image_array_clipped)
        dropped_counts = []

        array_values = image_array_clipped["et_ensemble_mad_index"].values
        array_values = array_values[~np.isnan(array_values)]
        dropped_count = [0, 0, 0, 0, 0, 0]
        for value in array_values:
            binary_string = format(int(value), "#08b")[2:]
            for k in range(len(binary_string)):
                if binary_string[k] == "0":
                    dropped_count[k] += 1
        dropped_count.append(len(array_values))
        print(dropped_count)
        dropped_counts.append(dropped_count)

        dropped_counts = np.array(dropped_counts)

        image_array_timeseries = image_array_clipped.mean(["x", "y"], skipna=True)

        image_array_df = image_array_timeseries.to_dataframe()
        image_array_df["DisALEXI_dropped"] = dropped_counts[:, 0]
        image_array_df["EEMETRIC_dropped"] = dropped_counts[:, 1]
        image_array_df["GEESEBAL_dropped"] = dropped_counts[:, 2]
        image_array_df["PTJPL_dropped"] = dropped_counts[:, 3]
        image_array_df["SIMS_dropped"] = dropped_counts[:, 4]
        image_array_df["SSEBop_dropped"] = dropped_counts[:, 5]
        image_array_df["Total_pixels"] = dropped_counts[:, 6]
        print("saving to ")
        print(base_output_dir + str(gsa_ids[i]) + "/" + image_key + ".csv")
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
# data['openet'] = [ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0'), "et_ensemble_mad", 1, 25000]
