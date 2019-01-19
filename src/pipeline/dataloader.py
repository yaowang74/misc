# -*- coding: utf-8 -*-
"""
Data loader is used to read raw data sets

@author: H211803
"""
import os.path
import errors as err
import pandas as pd
from loggerinitializer import InitializeLogger
from config import *

dl_logger = InitializeLogger('DataLoader')


class DataLoader:
    """

    """

    def __init__(self, raw_data_path=RAW_DATA_PATH):
        """

        :param raw_data_path: directory of raw data files
        """
        self.input_path = raw_data_path
        self.data_dict = {}
        dl_logger.info("A DataLoader object was initialized.\n")


    def _remove_pii(self):
        """remove those columns that contain personal identification
        information

        :return: same structure as self.data_dict, with PII columns removed
        from each data set.
        """
        if not self.data_dict:
            raise ValueError("There is no data in data_dict.\n")

        dl_logger.info("Start to remove PII columns ...\n")

        for key, val in self.data_dict.items():

            try:
                dl_logger.info("removing columns {0} from {1} ...\n".format(
                    EXCEL_PII_DICT[key], key))
                val.drop(columns=EXCEL_PII_DICT[key], inplace=True)
            except:
                dl_logger.debug("DataLoader._remove_pii")
                raise err.InvalidConfigError

        dl_logger.info("PII columns are removed from raw data.\n")

        return self


    def _load_flat_file(self, file_path, name, **kwargs):
        """load data stored in flat files, such as .csv or .tsv

        :param file_path: full path of the flat files
        :param name: key name of self.data_dict
        :param kwargs: keyworded arguments in pandas.read_csv
        :return: pandas dataframe
        """
        try:
            data = pd.read_csv(file_path, sep=kwargs['sep'],
                               encoding=kwargs['encoding'],
                               dtype=EXCEL_DTYPE_DICT[name],
                               parse_dates=DATETIME_COL_DICT[name])
        except:
            raise (err.NotFoundError)

        data.columns = data.columns.str.strip()

        self.data_dict.update({name: data})


    def _load_excel_file(self, file_path, name, **kwargs):
        """

        :param file_path: full path of excel file
        :param name: file name
        :param kwargs: keyworded arguments in pandas.read_excel
        :return: pandas dataframe
        """
        excel_file = pd.ExcelFile(io=file_path)

        for sheet_name in excel_file.sheet_names:
            tab = (name + "_" + sheet_name).strip().lower()

            try:
                data = pd.read_excel(
                    io=file_path, sheet_name=sheet_name, header=0,
                    dtype=EXCEL_DTYPE_DICT[tab],
                    na_values=EXTRA_NA_VALUES, keep_default_na=True)
            except:
                raise (err.NotFoundError)

            data.columns = data.columns.str.strip()

            self.data_dict.update({tab: data})


    def _load_raw_data(self):
        """load raw data from flat files and Excel spreadsheets.

        :return: a dictionary, each value of a key is a dataframe
        """
        dl_logger.info(
            "Start to load raw data from {} ...\n".format(RAW_DATA_PATH))

        file_list = os.listdir(RAW_DATA_PATH)

        if not file_list:
            raise ValueError(
                "There are no any data files in {}\n".format(RAW_DATA_PATH))

        for file in file_list:
            # assuming string after dot is the file type
            type = file.split('.')[-1].strip().lower()
            name = file.split('.')[0].strip().lower()
            file_path = os.path.join(RAW_DATA_PATH, file)

            dl_logger.info("Loading {} ...\n".format(file))

            if type == 'csv':
                kwargs = {'sep': ',', 'encoding': 'utf-8'}
                self._load_flat_file(file_path=file_path, name=name, **kwargs)

            elif type in ['tab', 'tsv']:
                kwargs= {'sep': '\t', 'encoding': 'cp1252'}
                self._load_flat_file(file_path=file_path, name=name, **kwargs)

            elif type == 'xlsx':

                self._load_excel_file(file_path=file_path, name=name)

            else:
                raise ValueError("{} file is not included yet.\n".format(type))

        dl_logger.info("Loaded data files are: {}.\n".format(file_list))
        # # output columns grouped by dtypes
        # for key, val in self.data_dict.items():
        #     dtype_dict = val.columns.to_series().groupby(val.dtypes).groups
        #
        #     dl_logger.info("dtypes of {}: \n".format(key))
        #
        #     for sub_key, sub_val in dtype_dict.items():
        #         print("{0}: {1}\n".format(sub_key, sub_val))

        dl_logger.info("finished loading data.\n")
        return self


    def load(self):
        """ entry point of this module

        :return:
        """
        self._load_raw_data()
        self._remove_pii()
