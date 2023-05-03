#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:38:49 2023

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
base_output_dir = "../data/cropland/"
cv_shape = gp.read_file("../shape/Alluvial_Bnd.shp")
cv_shape.to_crs(3310, inplace=True)
cv_shape["geometry"] = cv_shape.geometry.buffer(10000)


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
image_names = ["cropland"]
image_locations = [ee.ImageCollection("USDA/NASS/CDL")]
image_variables = [["cropland"]]
image_dict = {}

# for i in range(len(years) - 1):
#    for j in range(len(image_names)):
#        image_dict[image_names[j] + "_" + years[i]] = (
#            image_locations[j]
#            .select(image_variables[j])
#            .filterDate(years[i], years[i + 1])
#        )
image = (
    image_locations[0].select(image_variables[0]).filterDate("2000-01-01", "2022-01-01")
)
# for image_key in image_dict.keys():
#
#    # Load shapefile
#    print("Processing {}".format(image_key))
#
#    shp = cv_shape
#    shp_degree = shp.to_crs(4326)
#
#    image = image_dict[image_key]
#    bounding_box = shp_degree["geometry"].bounds
#    w_bounding_box = float(bounding_box["minx"]) - 0.01
#    s_bounding_box = float(bounding_box["miny"]) - 0.01
#    e_bounding_box = float(bounding_box["maxx"]) + 0.01
#    n_bounding_box = float(bounding_box["maxy"]) + 0.01
#    image_array = image.wx.to_xarray(
#        region=ee.Geometry.BBox(
#            w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box
#        ),
#        path=base_output_dir + image_key,
#    )

shp = cv_shape
shp_degree = shp.to_crs(4326)

bounding_box = shp_degree["geometry"].bounds
w_bounding_box = float(bounding_box["minx"]) - 2
s_bounding_box = float(bounding_box["miny"]) - 2
e_bounding_box = float(bounding_box["maxx"]) + 2
n_bounding_box = float(bounding_box["maxy"]) + 2

print(w_bounding_box)
# middle_ns = (n_bounding_box + s_bounding_box) / 1000
# middle_we = (w_bounding_box + e_bounding_box) /
distance_ns = abs(n_bounding_box - s_bounding_box) / 5
distance_we = abs(w_bounding_box - e_bounding_box) / 5
# image_array.rio.write_crs(4326, inplace=True)
for j in range(1, 5):
    new_distance_s = distance_ns * (j - 1)
    new_distance_n = distance_ns * j
    for i in range(1, 5):
        new_distance_w = distance_we * (i - 1)
        new_distance_e = distance_we * i

        image_array = image.wx.to_xarray(
            # region=ee.Geometry.BBox(w_bounding_box, middle_ns, middle_we, n_bounding_box),
            # region=ee.Geometry.BBox(
            #    w_bounding_box, s_bounding_box, e_bounding_box, n_bounding_box
            # ),
            region=ee.Geometry.BBox(
                w_bounding_box + new_distance_w,
                s_bounding_box + new_distance_s,
                w_bounding_box + new_distance_e,
                s_bounding_box + new_distance_n,
            ),
            path=base_output_dir
            + "cropland_2000_2020_test_3_{i}_{j}.nc".format(i=i, j=j),
            scale=250,
        )


"""
def preprocess(ds):
    return ds.drop_duplicates(dim=["x", "y", "time"])


def drop_duplicates_along_all_dims(obj, keep="first"):
    all_dims = obj.dims
    indexes = {dim: ~obj.get_index(dim).duplicated(keep=keep) for dim in all_dims}
    return obj.isel(indexes)


ds = xr.open_mfdataset("../data/cropland/cropland_*.nc")
# combine="nested",
# concat_dim="time",
# )
ds.to_netcdf("~/Desktop/cropland_combined_2.nc")
"""
