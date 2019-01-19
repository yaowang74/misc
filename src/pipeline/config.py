# -*- coding: utf-8 -*-
"""
Application settings and configurations

@author: H211803
"""
import os.path
import numpy as np

## data path settings
BASE_DIR = "/home/h211803/BitBucket/supplier_decommit_scm"

# raw data
RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw")

# processed data
PROC_DATA_PATH = os.path.join(BASE_DIR, "data/proc")

# models save path
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models/")

## TASK_TYPE: 'TRAIN', 'TRAIN_AND_SCORE', 'SCORE'
## PREDICT_DECOMMIT_WEEK_Y_N: boolean, default is True, if False,
## just predict whether a PO will decommit.
TASK_TYPE = 'TRAIN_AND_SCORE'
PREDICT_DECOMMIT_WEEK_Y_N = True

## Per customer's requirment, prediction need to be performed 13 weeks
## before requested delivery date.
BLACKOUT_WEEKS = 13

## H2o local cluster setttings
NTHREADS = -1
MAX_MEM_SIZE = 32

## Sampling factor, a value in (0, 1) and random seed
DOWNSAMPLE_FACTOR = 0.1
RANDOM_SEED = 74

# string for good PO
NO_DECOMMIT = 'NO DECOMMIT'

## Excel data type schema
PO_CHANGE_HISTORY_SHEET1_DTYPE = {
    "ID" : str,
    "ImportHistoryID" : str,
    "PONumber" : str,
    "POItemNo" : str,
    "Schedulelinenumber" : str,
    "Header_ItemChange"	: str,
    "ShortText" : str,
    "NewValue" : str,
    "OldValue" : str,
    "UserName" : str,
    "UserEID" : str,
    "Versionnumber" : np.int16,
    "ChangeDateTime" : np.dtype('datetime64[ns]'),
    "Vendor" : str,
    "PurchasingGroup" : str,
    "PartNumber" : str,
    "Plant" : str,
    "VendorName" : str,
    "vendor2" : str
}

RECEIPTS_SHEET1_DTYPE = {
    "SBG_ID": str,
    "SBE_ID": str,
    "SITE_ID": str,
    "SITE_NAME": str,
    "PART_NUMBER": str,
    "PART_DESC": str,
    "SUPPLIER_ID": str,
    "SUPPLIER_NAME": str,
    "SITE_LOCAL_SUPPLIER_NUMBER": str,
    "SITE_LOCAL_SUPPLIER_NAME": str,
    "SCO": str,
    "CO_CST": str,
    "YEAR": np.int16,
    "MONTH": str,
    "SCHEDULED_DELIVERY_DATE": np.dtype('datetime64[ns]'),
    "REQUESTED_DELIVERY_DATE": np.dtype('datetime64[ns]'),
    "ACTUAL_DELIVERY_DATE": np.dtype('datetime64[ns]'),
    "SPEND_TYPE": str,
    "ORDERED_QTY": np.int16,
    "RECEIVED_QTY": np.int16,
    "RECEIPT_NUMBER": str,
    "PO_NUMBER": str,
    "PO_LINE_NUMBER": str,
    "PART_COMMODITY_FAMILY": str,
    "PART_COMMODITY_FAMILY_DESC": str,
    "UNIT_OF_MEASURE": str,
    "HISTORICAL_CONTRACT_ID": str,
    "CURRENT_CONTRACT_ID": str,
    "PART_PLANNER_BUYER": str,
    "RCPT_BUYER_NAME": str,
    "SITE_LEAD_TIME": np.int16,
    "PO_TERMS": str,
    "PO_NET_DAYS": object,
    "PART_PFC": str,
    "PART_CST": str,
    "UNSPSC_PART": str,
    "PART_LOB": str,
    "ENTERPRISE_HOS_GOLD": str,
    "PART_COE_CODE": str,
    "CID": str,
    "CID_LAST_CODED_DATE": str,
    "CID_LAST_CODED_NAME": str,
    "COMPLIANCE_COMMENTS": str,
    "EXPORT_REVIEW": str,
    "SITE_SOURCING_ATTRIBUTE_1": str,
    "SITE_SOURCING_ATTRIBUTE_2": str,
    "SITE_SOURCING_ATTRIBUTE_3": str,
    "SITE_SOURCING_ATTRIBUTE_4": str,
    "SOURCE_CONTROL": str,
    "PART_SEGMENT_CODE": str,
    "PULL_CANDIDATE_Y_N": str,
    "ARS_STRATEGY": str,
    "ARS_STRATEGY_DESC": str,
    "AUTO_PO_Y_N": str,
    "CONSIGNMENT_Y_N": str,
    "REPLENISH_TYPE": str,
    "DEMAND_PFEP": str,
    "PART_REVISION": str,
    "TARGET_COST": np.float32,
    "ASSUMPTIONS_TC": str,
    "LAST_CHANGE_DATE_TC": np.dtype('datetime64[ns]'),
    "TARGET_COST_CURRENCY": str,
    "SIMILAR_PART": str,
    "AER_STRATEGY": str,
    "ACS_STRATEGY": str,
    "PMT_STRATEGY": str,
    "TS_STRATEGY": str,
    "HDQ_STRATEGY": str,
    "AER_COE_SUPPLIER": str,
    "ACS_COE_SUPPLIER": str,
    "PMT_COE_SUPPLIER": str,
    "TS_COE_SUPPLIER": str,
    "SUPPLIER_UNSPSC_CODE": str,
    "SUPPLIER_UNSPSC_DESC": str,
    "SUPPLIER_GEOGRAPHY": str,
    "SUPPLIER_COUNTRY": str,
    "INTER_COMPANY_Y_N": str,
    "SQP_Y_N": str,
    "CURRENCY_CODE": str,
    "CONTRACT_PRICE_LOWEST": np.float32,
    "BASELINE_UNIT_PRICE_LOCAL_CY": np.float32,
    "BASELINE_UNIT_PRICE": np.float32,
    "ACTUAL_UNIT_PRICE_LOCAL_CY": np.float32,
    "ACTUAL_UNIT_PRICE": np.float32,
    "CONTRACT_UNIT_PRICE_LOCAL_CY": np.float32,
    "CONTRACT_UNIT_PRICE": np.float32,
    "BASELINE_AMOUNT_LOCAL_CY": np.float32,
    "BASELINE_AMOUNT": np.float32,
    "SPEND_LOCAL_CY": np.float32,
    "SPEND": np.float32,
    "CONTRACT_AMOUNT_LOCAL_CY": np.float32,
    "CONTRACT_AMOUNT": np.float32,
    "COST_SAVINGS_LOCAL_CY": np.float32,
    "COST_SAVINGS": np.float32,
    "EXPECTED_PO_VS_AC_AMT_LOCAL_CY": np.float32,
    "EXPECTED_PO_VS_ACTUAL_AMT": np.float32,
    "CONTRACT_VS_ACT_AMT_LOCAL_CY": np.float32,
    "CONTRACT_VS_ACTUAL_AMT": np.float32,
    "PO_PLACE_DATE": np.dtype('datetime64[ns]'),
    "ACTUAL_LEAD_TIME": np.int16,
    "AWPOT_NUMERATOR": np.float32,
    "AWPOT_DENOMINATOR": np.float32,
    "SUPPLIER_GROUP_ID": str,
    "SUPPLIER_GROUP_NAME": str,
    "PO_LINE_DESCRIPTION": str,
    "MANUF_SUPP_NAME": str,
    "MANUF_SUPP_PART": str,
    "HYPERION_CODE": str,
    "UNSPSC": str,
    "CHARGE_ACCOUNT": str,
    "COUNTRY_OF_ORIGIN": str,
    "PRODUCT_LINE": str,
    "ELECTRONIC_PURCHASE": str,
    "SBE_NAME": str,
    "SITE_REGION": str,
    "BUYHON_CLASSIFICATION": str,
    "BUYHON_CONFIDENCE_LEVEL": np.float32,
    "BUYHON_LAST_MOD_DATE": np.dtype('datetime64[ns]'),
    "Plant": str,
    "Matnr": str,
    "material": object,
    "Part": str

}

DECOMMITS_DECOMMIT_DTYPE = {
    "ID": str,
    "SCOE": str,
    "PONum": str,
    "POItem": str,
    "Schedulelinenumber": str,
    "ShortText": str,
    "NewCommitDate": np.dtype('datetime64[ns]'),
    "OriginalCommitDate": np.dtype('datetime64[ns]'),
    "Vendor": str,
    "Date": np.dtype('datetime64[ns]'),
    "VendorName": str,
    "PurchasingGroup": str,
    "BuyerName": str,
    "Manager": str,
    "Dir": str,
    "SrDir": str,
    "AEROCOE": str,
    "PartNumber": str,
    "Plant": str,
    "Week": np.int16,
    "ReportDate": np.dtype('datetime64[ns]'),
    "NOC": np.int16,
    "MOC": np.float32,
    "NotificationofChange": str,
    "MagnitudeofChange": str,
    "ExtendedCost": np.float32,
    "Contract": str,
    "PFEP": str,
    "PurchaseTextNote": str,
    "Lead_Time": object,
    "Days": np.int16,
    "ShorttoLead": str,
    "StatDelDate": np.dtype('datetime64[ns]'),
    "FirstDecommit": str,
    "IDO/NONIDO": str,
    "Concat": str
}
# PD Commits are "NO Shows", means "definite decommit"
DECOMMITS_PD_COMMIT_DTYPE = DECOMMITS_DECOMMIT_DTYPE

GIVRRDATALOAD_DTYPE = {
    "Part Number": str,
    "Site": str,
    "Part Family ": str,
    "Saf Stk": np.dtype('int32'),
    "OHB": np.dtype('int32'),
    "OHB$": np.dtype('float16'),
    "Past Due Dollars": np.dtype('float16'),
    "12 Mo Req": np.dtype('int32'),
    "Std Unit Cost": np.dtype('float16'),
    "Planner Name": str,
    "OverMaxthruLT": np.dtype('int32'),
    "UnderMinthruLT": np.dtype('float16'),
    "Product Hierarchy": str,
    "PFEP": str,
    "Cold Start Lead Time": np.dtype('int16'),
    "Supply Type": str,
    "Valuation Class": str,
    "Warm Start LT": np.dtype('int32'),
    "Scheduled Receipts Issue Date": object,
    "Replenishment Lead Time": np.dtype('int32'),
    "WIP": np.dtype('int32'),
    "WIP Dollars": np.dtype('int32'),
    "90 Day His": np.dtype('int16'),
    "12 Month His": np.dtype('int16'),
    "Date": object,
    "Reserve$": np.dtype('float16'),
    "PFEP Supply": str,
    "Deliv LT": np.dtype('int32'),
    "TtlReplLT": np.dtype('int32'),
    "Valuation Type": str,
    "Past Due Demands": np.dtype('int32'),
    "Buyer Code Value": str,
    "Buyer Code Description": str,
    "Avg Days Late": np.dtype('int32'),
    "ABC Code": str,
    "EOQ": np.dtype('int32'),
    "Procure Method": str,
    "Planner Code": str,
    "Production Scheduler": str,
    "Demand Type": str,
    "Catalog Lead Time": np.dtype('int32')
}

EXCEL_DTYPE_DICT = {
    'po change history_sheet1': PO_CHANGE_HISTORY_SHEET1_DTYPE,
    'receipts_sheet1': RECEIPTS_SHEET1_DTYPE,
    'decommits_decommit': DECOMMITS_DECOMMIT_DTYPE,
    'decommits_pd commit' : DECOMMITS_PD_COMMIT_DTYPE,
    'givrrdataload': GIVRRDATALOAD_DTYPE
}
# strings in this list will be treated as missing values
EXTRA_NA_VALUES = ['00/00/0000']

# columns names that are of datetime type
DATETIME_COL_DICT = {
    'po change history_sheet1': [],
    'receipts_sheet1': [],
    'decommits_decommit': [],
    'decommits_pd commit': [],
    'givrrdataload': ['Scheduled Receipts Issue Date', 'Date']
}

## columns of PII, remove 'BuyerName' from this list in case we need it
DECOMMITS_DECOMMIT_PII_LIST = ['ID', 'BuyerName', 'Manager', 'Dir', 'SrDir']

DECOMMITS_PD_COMMIT_PII_LIST = DECOMMITS_DECOMMIT_PII_LIST

PO_CHANGE_HISTORY_SHEET1_PII_LIST = ['ID', 'ImportHistoryID', 'UserName']

RECEIPTS_SHEET1_PII_LIST = [
    'SITE_NAME', 'PART_DESC', 'PART_PLANNER_BUYER', 'RCPT_BUYER_NAME',
    'CID_LAST_CODED_NAME', 'SITE_SOURCING_ATTRIBUTE_4','ACS_STRATEGY',
    'PMT_STRATEGY', 'TS_STRATEGY', 'HDQ_STRATEGY', 'ACS_COE_SUPPLIER',
    'PMT_COE_SUPPLIER', 'TS_COE_SUPPLIER', 'CHARGE_ACCOUNT', 'SBE_NAME',
    'Matnr', 'material', 'Part', 'PART_COMMODITY_FAMILY',
    'PART_COMMODITY_FAMILY_DESC']

MM_PII_LIST = ['Planner Name', 'Buyer Code Value', 'Buyer Code Description']

EXCEL_PII_DICT = {
    'po change history_sheet1': PO_CHANGE_HISTORY_SHEET1_PII_LIST,
    'receipts_sheet1': RECEIPTS_SHEET1_PII_LIST,
    'decommits_decommit': DECOMMITS_DECOMMIT_PII_LIST,
    'decommits_pd commit': DECOMMITS_PD_COMMIT_PII_LIST,
    'givrrdataload': MM_PII_LIST
}

## feature engine data key dictionary
FE_KEY_PAIR_DICT = {
    "po_change": 'po change history_sheet1',
    "receipts": 'receipts_sheet1',
    "decommit": 'decommits_decommit',
    "pd_commit": 'decommits_pd commit',
    "mm": 'givrrdataload'
}

## modling parameters, be careful: num_class is hard coded here.
params = {
    'task': 'train',
	'application': 'multiclass',
    'num_class': 14,
    'metric': 'multi_logloss'
}