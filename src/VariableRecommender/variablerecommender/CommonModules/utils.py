# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:10:01 2018

@author: H211803
"""

import os
import numpy as np
import pandas as pd
import configparser as cp
from CommonModules.dataloader import DataLoader
from CommonModules.loggerinitializer import InitializeLogger

utils_logger = InitializeLogger(app_name='Utils')


def ParentDir(_path, n_levels):
    """Climb upper levels of directory

    """
    return os.sep.join(_path.split(os.sep)[:-n_levels])


def ConfigReader(config_file_name='app_config.ini'):
    """Read config file

    """
    config = cp.ConfigParser()
    config.read(os.path.join(ParentDir(__file__, 5),
                             'config', config_file_name))

    return config


def FeatureInit():
    """Converting column types using user specs

    """
    cfg_content = ConfigReader('app_config.ini')

    data_loader = DataLoader(data_source=cfg_content['Input']['source'],
                             data_format=cfg_content['Input']['format'],
                             data_path=cfg_content['Input']['path'],
                             data_file_name=cfg_content['Input']['filename'],
                             uid=cfg_content['Input']['uid'],
                             password=cfg_content['Input']['password'],
                             odbc_string=cfg_content['Input']['odbc_string'],
                             hive_ql=cfg_content['Input']['hive_ql'])

    utils_logger.info("Start to read file.")
    data_loader.load()
    input_df = data_loader.data
    utils_logger.info("Data file was loaded.")
    utils_logger.info("First 5 rows of loaded dataframe are:")
    utils_logger.info(input_df.head())
    utils_logger.info("Datafrme dtypes:")
    utils_logger.info(input_df.dtypes)

    aux_loader = DataLoader(
            data_source=cfg_content['Auxiliary']['source'],
            data_format=cfg_content['Auxiliary']['format'],
            data_path=cfg_content['Auxiliary']['path'],
            data_file_name=cfg_content['Auxiliary']['filename'],
            uid=cfg_content['Input']['uid'],
            password=cfg_content['Input']['password'],
            odbc_string=cfg_content['Input']['odbc_string'],
            hive_ql=cfg_content['Input']['hive_ql'])
    utils_logger.info("Start to load auxiliary file for dtypes:")
    aux_loader.load()
    aux_df = aux_loader.data
    utils_logger.info("Auxiliary file was loaded.")
    utils_logger.info("Column types are:")
    utils_logger.info(aux_df)
    utils_logger.info("Start to convert column types according using " +
                      "auxiliary file")

    for i, row in aux_df.iterrows():
        if row['col_type'].lower() in ["categorical", "category"]:
            input_df[row['col_name']] =\
                input_df[row['col_name']].astype("category")
        elif row['col_type'].lower() == "integer":
            input_df[row['col_name']] =\
                input_df[row['col_name']].astype("int64")
        elif row['col_type'].lower() == "float":
            input_df[row['col_name']] =\
                input_df[row['col_name']].astype("float64")
        elif row['col_type'].lower() == "string":
            input_df[row['col_name']] =\
                input_df[row['col_name']].astype("object")
        elif row['col_type'].lower() == "datetime":
            input_df[row['col_name']] =\
                pd.to_datetime(input_df[row['col_name']])

    utils_logger.info("Column types convertion is finished")
    utils_logger.info("New column types are:")
    utils_logger.info(input_df.dtypes)

    return input_df, cfg_content


def ColumnTypeSelector(input_df):
    """Extract sub dataframe by column types

    """
    utils_logger.info("Extracting numeric columns...")
    num_sub_df = input_df.select_dtypes(include=np.number)
    if not num_sub_df.empty:
        utils_logger.info("First 5 rows of numeric columns:")
        utils_logger.info(num_sub_df.head())
    else:
        utils_logger.info("There are no numeric columns.")
    utils_logger.info("Extracting categorical columns...")
    cat_sub_df = input_df.select_dtypes(include='category')
    if not cat_sub_df.empty:
        utils_logger.info("First 5 rows of categorical columns:")
        utils_logger.info(cat_sub_df.head())
    else:
        utils_logger.info("There are no categorical columns.")
    return num_sub_df, cat_sub_df
