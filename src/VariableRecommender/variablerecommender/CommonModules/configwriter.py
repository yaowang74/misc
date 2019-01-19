# -*- coding: utf-8 -*-
"""
configuration file writer

@author: H211803
"""
import configparser as cp
import pandas as pd
# %%
REPO_PATH = "C:/Users/H211803/Documents/BitBucketRepository/smart_energy/src/"
PACKAGE_PATH = "SupervisedSegmentation/supervisedsegmentation/"
CONFIG_PATH = REPO_PATH + PACKAGE_PATH
CONFIG_FILE = "app_config.ini"
REGRESSION_AUXILIARY_FILE = 'regression_sample_config.csv'
CLASSIFICATION_AUXILIARY_FILE = 'classification_sample_config.csv'
REGRESSION_TREE_AUXILIARY_FILE = 'regression_tree_sample_config.csv'

# %%
"""Application config file writer

"""

config = cp.ConfigParser()
# filename in ['Input'] and ['Auxiliary'] must be matched
config['Input'] = {
        'source': 'local',
        'format': 'pickle',
        'path': "C:/Users/H211803/Documents/BitBucketRepository/smart_energy/data/",
        'filename': "REX_LP_noRev_20150101to20180225_siteDailyagg_50miss_prop_cdd66.pkl",
        'uid': 'base64 encoded UID',
        'password': 'base64 encoded PASSWORD',
        'odbc_string': 'DRIVER={Hortonworks Hive ODBC Driver};' +
                       'SERVER=Hive Server 2;' +
                       'Host=sal04sbx.azurehdinsight.net;' +
                       'DATABASE=default;port=443;' +
                       'Thrift Transport=HTTP;' +
                       'ssl=1;Authmech=6;',
        'hive_ql': "select *  from default.hvac limit 10;",
        'target': 'kwh'}

config['Output'] = {
        'path': "C:/Users/H211803/Documents/BitBucketRepository/smart_energy/output/",
        'filename': "VariableRecommender.txt"}
# filename in ['Input'] and ['Auxiliary'] must be matched
config['Auxiliary'] = {
        'source': 'local',
        'format': 'csv',
        'path': 'C:/Users/H211803/Documents/BitBucketRepository/smart_energy/src/SupervisedSegmentation/supervisedsegmentation/',
        'filename': 'regression_tree_sample_config.csv'}

config['Algorithm'] = {
        'DecisionTreeRegressor': "True",
        'DecisionTreeClassifier': "False"}

config['AlgoProperties'] = {'max_depth': '4',
                            'min_samples_split': '0.15',
                            'min_samples_leaf': '0.075',
                            'min_impurity_decrease': '0.001',
                            'random_state': '74',
                            'var_in_use':
                                ['has_pool', 'has_central_ac',
                                 'is_gas_user', 'propType', 'propSizes',
                                 'propAges', 'propValues'],
                            'groups_mean_diff': '9',
                            'prune_threshold': '0'}

with open(CONFIG_PATH + CONFIG_FILE, 'w') as configfile:
    config.write(configfile)

# %%
"""Regression tree data auxiliary CSV file writer

"""
col_type_dict = {'col_name': ['site_name', 'date', 'kwh',
                              'has_pool',
                              'has_central_ac',
                              'is_gas_user',
                              'propType',
                              'propSizes',
                              'propAges',
                              'propValues',
                              'cdd66'],
                 'col_type': ['string', 'datetime', 'float',
                              'category', 'category', 'category',
                              'category', 'category', 'category',
                              'category', 'integer']}
pd.DataFrame(data=col_type_dict).to_csv(CONFIG_PATH +
                                        REGRESSION_TREE_AUXILIARY_FILE,
                                        index=False)

# %%
"""Regression data auxiliary CSV file writer

"""
col_type_dict = {'col_name': ['meter_name', 'kwh', 'date',
                              'min_temperature_windchill_2m_f',
                              'max_pressure_2m_mb',
                              'min_wind_speed_10m_mph',
                              'tot_precipitation_in',
                              'tot_snowdepth_in',
                              'avg_cloud_cover_tot_pct',
                              'weekend_ind'],
                 'col_type': ['string', 'float', 'datetime',
                              'float', 'float', 'float',
                              'float', 'float', 'float',
                              'category']}
pd.DataFrame(data=col_type_dict).to_csv(CONFIG_PATH +
                                        REGRESSION_AUXILIARY_FILE, index=False)

# %%
"""Classification data auxiliary CSV file writer

"""
col_type_dict = {'col_name': ['mtr_name', 'date', 'is_theft', 'avg_actl_use_7',
                              'e_use7', 'engy_use_chng', 'use_dif7',
                              'zero_use_ind', 'drop_ind', 'spike_ind',
                              'is_dif7_more_98pctl', 'is_dif7_less_2pctl',
                              'tilt_warning', 'Time.Changed',
                              'mtr_report_load_side_volt',
                              'load_side_volt', 'reverse_engy',
                              'interval_data_gap', 'interval_outage',
                              'interval_restor', 'disconn_relay_open',
                              'latched_load_side_volt_prsnt',
                              'load_side_volt_prsnt', 'tilt_warning_status',
                              'reverse_power_warning', 'volt_on_disconn',
                              'power_status_unknown', 'sustained_outage',
                              'momentary_outage', 'alert_num'],
                 'col_type': ['string', 'datetime', 'category', 'float',
                              'float', 'float', 'float','category', 'category',
                              'category', 'category', 'category', 'integer',
                              'integer', 'integer', 'integer', 'integer',
                              'integer', 'integer', 'integer', 'integer',
                              'integer', 'integer', 'integer', 'integer',
                              'integer', 'integer', 'integer', 'integer',
                              'integer']}
pd.DataFrame(data=col_type_dict).to_csv(CONFIG_PATH +
                                        CLASSIFICATION_AUXILIARY_FILE,
                                        index=False)
