# -*- coding: utf-8 -*-
"""
Data loader

@author: H211803
"""

import os
import csv
import base64
# import pyodbc
import pandas as pd
import traceback as tb
import CommonModules.errors as err
from CommonModules.loggerinitializer import InitializeLogger

data_loader_logger = InitializeLogger('DataLoader')

SUPPORTED_DATA_SOURCE = ['local', 'sentience']
SUPPORTED_DATA_FORMAT = ['hive', 'csv', 'pickle', 'xml', 'json']


class DataLoader:
    """

    """

    def __init__(self, data_source, data_format, data_path, data_file_name,
                 uid, password, odbc_string, hive_ql):
        """

        """
        self.source = data_source
        self.format = data_format
        self.path = data_path
        self.file = data_file_name
        if data_source.lower == "sentience":
            self.uid = base64.b64decode(uid).decode()
            self.password = base64.b64decode(password).decode()

        self.odbc_str = odbc_string
        self.hive_ql = hive_ql
        self.data = pd.DataFrame()

    def load(self):
        """

        """
        if self.source.lower() == 'local':
            if self.format.lower() == 'csv':
                try:
                    self.data = pd.read_csv(os.path.join(self.path, self.file))
                except Exception as e:
                    data_loader_logger.error("Input CSV File Not Found")
                    data_loader_logger.info(tb.format_exc())
                    raise err.NotFoundError("Data specified in configuraton" +
                                            " file was not found.")
                if self.data.empty:
                    data_loader_logger.info("pandas.DataFrame.empty is True")
            elif self.format.lower() == 'json':
                data_loader_logger.info("Loading json file...")
                try:
                    self.data = pd.read_json(
                            os.path.join(self.path, self.file))
                except Exception as e:
                    data_loader_logger.error("Input json File Not Found")
                    data_loader_logger.info(tb.format_exc())
                    raise err.NotFoundError("Data specified in configuraton" +
                                            " file was not found.")
            elif self.format.lower() == 'xml':
                try:
                    with open(os.path.join(self.path, self.file)) as f:
                        f.close()
                except Exception as e:
                    data_loader_logger.error("Input xml File Not Found")
                    data_loader_logger.info(tb.format_exc())
                    raise err.NotFoundError("Data specified in configuraton" +
                                            " file was not found.")
            elif self.format.lower() == 'pickle':
                data_loader_logger.info("Loading pickle file...")
                try:
                    self.data = pd.read_pickle(
                            os.path.join(self.path, self.file))
                except Exception as e:
                    data_loader_logger.error("Input pickle File Not Found")
                    data_loader_logger.info(tb.format_exc())
                    raise err.NotFoundError("Data specified in configuraton" +
                                            " file was not found.")
        elif self.source.lower() in 'sentience':
            if self.format.lower() == 'hive':
                pass
# =============================================================================
#                 odbc_str = (self.odbc_str + 'UID=' + self.uid +
#                             '@sentience.local;PWD='+self.password)
#                 try:
#                     cnxn = pyodbc.connect(odbc_str, autocommit=1)
#                     cursor = cnxn.cursor()
#                     rows = cursor.execute(self.hive_ql)
#                     with open(r'C:\\hive.csv', 'w', newline='') as csvfile:
#                         writer = csv.writer(csvfile)
#                         writer.writerow([x[0] for x in cursor.description])
#                         for row in rows:
#                             writer.writerow(row)
# 
#                     self.data = pd.read_csv('C:/hive.csv')
#                     os.remove('C:\\hive.csv')
#                 except Exception as e:
#                     data_loader_logger.error("Could not load HIVE table...")
#                 finally:
#                     cursor.close()
#                     cnxn.close()
# =============================================================================
        return self
