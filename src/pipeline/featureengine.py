# -*- coding: utf-8 -*-
"""
Basic data cleansing and feature engineering.

@author: H211803
"""
import os.path
import utils as ut
import errors as err
import numpy as np
import pandas as pd
from loggerinitializer import InitializeLogger
from config import PROC_DATA_PATH, FE_KEY_PAIR_DICT, NO_DECOMMIT
from config import BLACKOUT_WEEKS


fe_logger = InitializeLogger('FeatureEngine')


class FeatureEngine:
    """data cleansing, feature creation etc.

    """

    def __init__(self, dl):
        """

        :param dl: a DataLoader object with non-empty data_dict
        """
        fe_logger.info("Initializing feature engine to transform raw data. \n")
        if isinstance(dl.data_dict, dict):
            if not dl.data_dict:
                raise err.InputError

            self.po_change = dl.data_dict[FE_KEY_PAIR_DICT['po_change']].copy()
            self.decommit = dl.data_dict[FE_KEY_PAIR_DICT['decommit']].copy()
            self.pd_commit = dl.data_dict[FE_KEY_PAIR_DICT['pd_commit']].copy()
            self.receipts = dl.data_dict[FE_KEY_PAIR_DICT['receipts']].copy()
            self.mm = dl.data_dict[FE_KEY_PAIR_DICT['mm']].copy()

            fe_logger.info(
                "Extracting PO Numbers from PO Change "
                "History as population that either have stat date changed or "
                "delivery date changed ... \n")

            cond = (self.po_change['ShortText'] == 'ItemDeliveryDate') | \
                   (self.po_change['ShortText'] ==
                    'Statistics-RelevantDeliveryDate')
            po_change_po = self.po_change[cond]['PONumber'].copy()

            po_change_po = set(po_change_po)

            # fe_logger.info("Filtering those PO Numbers that already closed "
            #                "so that requested deliver date are not null ...\n")
            #
            # receipts_po = set(self.receipts['PO_NUMBER'])
            #
            # self.po_population = pd.Series(
            #     list(po_change_po.intersection(receipts_po)))

            # AS OF JAN, 2019 WE HAVE NOT REALLY USED VARIABLES FROM RECEIPTS,
            # WE DO NOT FILTER USE PO NUMBERS IN RECEIPTS DATA.
            self.po_population = set(po_change_po)

            self.po_change = self.po_change.loc[
                             self.po_change['PONumber'].isin(
                self.po_population), :]

            fe_logger.info("Extracting decommit data for PO population ... \n")

            self.decommit = self.decommit.loc[self.decommit['PONum'].isin(
                self.po_population), :]

            self.pd_commit = self.pd_commit.loc[self.pd_commit['PONum'].isin(
                self.po_population), :]

            self.receipts = self.receipts.loc[self.receipts['PO_NUMBER'].isin(
                self.po_population), :]

            fe_logger.info("Extracting part number from PO population ... \n")

            part_number_population = self.po_change.loc[:, 'PartNumber'].copy()
            self.part_number_population = \
                part_number_population.drop_duplicates()
            self.mm = self.mm.loc[self.mm['Part Number'].isin(
                self.part_number_population), :]
        else:
            raise err.InputError

    def _process_base(self):
        """construct a base data frame. This base data frame will be
        appended more columns and built to train predictive model.

        :return: data frame
        """
        fe_logger.info("Constructing a base data frame.\n")
        drop_col = ['Header_ItemChange', 'ShortText', 'NewValue',
                    'OldValue', 'UserEID', 'Versionnumber',
                    'ChangeDateTime']

        self.base = self.po_change.drop(columns=drop_col)
        self.base = self.base[(self.base['Schedulelinenumber'] != 'nan')]
        self.base = self.base.loc[self.base['PONumber'].isin(
            self.po_population), :]
        self.base = self.base.drop_duplicates()
        self.base = self.base.reset_index(drop=True)
        cat_col = ['Vendor', 'PurchasingGroup', 'PartNumber', 'Plant',
                   'VendorName', 'vendor2']
        self.base = ut.unified_dtype_conversion(self.base, cat_col, 'category')

        fe_logger.info("Columns in base dataframe are: {}.\n".format(
            self.base.columns.values.tolist()))
        return self

    def _process_receipts(self):
        """process receipts data, aggregate some columns. NOTE that there is
        scheduled line number in receipt data. PO line number is actual PO
        item number.

        :return: data frame
        """

        def get_trans_settings():
            """simply return group by columns list and dictionary that will be
            used groupby.agg and list of redundant columns list

            :return: group by column list and aggregation dictionary
            """
            group_by = ['PO_NUMBER', 'PO_LINE_NUMBER', 'PART_NUMBER',
                        'Plant', 'SITE_LOCAL_SUPPLIER_NUMBER']

            agg_dict = {
                'SUPPLIER_ID': 'first',
                'SUPPLIER_NAME': 'first',
                'SITE_LOCAL_SUPPLIER_NAME': 'first',
                'SCO': 'first',
                'CO_CST': 'first',
                'SCHEDULED_DELIVERY_DATE': 'first',
                'REQUESTED_DELIVERY_DATE': 'first',
                'ACTUAL_DELIVERY_DATE': 'first',
                'ORDERED_QTY': np.sum,
                'RECEIVED_QTY': np.sum,
                'SITE_LEAD_TIME': 'first',
                'PO_NET_DAYS': 'first',
                'CID': 'first',
                'CID_LAST_CODED_DATE': 'first',
                'DEMAND_PFEP': 'first',
                'TARGET_COST': np.sum,
                'AER_COE_SUPPLIER': 'first',
                'SUPPLIER_UNSPSC_CODE': 'first',
                'SUPPLIER_UNSPSC_DESC': 'first',
                'SUPPLIER_GEOGRAPHY': 'first',
                'SUPPLIER_COUNTRY': 'first',
                'CONTRACT_PRICE_LOWEST': 'first',
                'BASELINE_UNIT_PRICE': 'first',
                'ACTUAL_UNIT_PRICE': 'first',
                'CONTRACT_UNIT_PRICE': 'first',
                'BASELINE_AMOUNT': np.sum,
                'SPEND': np.sum,
                'CONTRACT_AMOUNT_LOCAL_CY': np.sum,
                'CONTRACT_AMOUNT': np.sum,
                'COST_SAVINGS': np.sum,
                'CONTRACT_VS_ACTUAL_AMT': np.sum,
                'PO_PLACE_DATE': 'first',
                'ACTUAL_LEAD_TIME': 'first',
                'AWPOT_NUMERATOR': 'first',
                'AWPOT_DENOMINATOR': 'first',
                'SUPPLIER_GROUP_ID': 'first',
                'SUPPLIER_GROUP_NAME': 'first',
                'PO_LINE_DESCRIPTION': 'first',
                'MANUF_SUPP_NAME': 'first',
                'MANUF_SUPP_PART': 'first',
                'PRODUCT_LINE': 'first',
                'ELECTRONIC_PURCHASE': 'first',
                'BUYHON_CLASSIFICATION': 'first',
                'BUYHON_CONFIDENCE_LEVEL': 'first',
                'BUYHON_LAST_MOD_DATE': 'first',
                'BLACKOUT_DATE': 'first',
                'DELTA_DAYS_ACTL_REQ_DEL_DATE': np.sum,
                'PLANNED_LEAD_DAYS': 'first',
                'DELTA_QTY_ORDERED_RECEIVED': np.sum}

            drop_col = [
                'SITE_ID', 'SBG_ID', 'SBE_ID', 'RECEIPT_NUMBER', 'YEAR',
                'MONTH', 'HISTORICAL_CONTRACT_ID', 'CURRENT_CONTRACT_ID',
                'SITE_SOURCING_ATTRIBUTE_1', 'SITE_SOURCING_ATTRIBUTE_2',
                'SITE_SOURCING_ATTRIBUTE_3', 'AER_STRATEGY', 'ARS_STRATEGY',
                'ARS_STRATEGY_DESC', 'SPEND_TYPE', 'AUTO_PO_Y_N',
                'CONSIGNMENT_Y_N', 'REPLENISH_TYPE', 'ASSUMPTIONS_TC',
                'INTER_COMPANY_Y_N', 'PART_COE_CODE', 'ASSUMPTIONS_TC',
                'EXPORT_REVIEW', 'SOURCE_CONTROL', 'COMPLIANCE_COMMENTS',
                'PART_PFC', 'PART_CST', 'UNSPSC_PART', 'PART_LOB',
                'ENTERPRISE_HOS_GOLD', 'LAST_CHANGE_DATE_TC',
                'TARGET_COST_CURRENCY', 'SIMILAR_PART', 'CURRENCY_CODE',
                'EXPECTED_PO_VS_AC_AMT_LOCAL_CY', 'SITE_REGION',
                'EXPECTED_PO_VS_ACTUAL_AMT', 'CONTRACT_VS_ACT_AMT_LOCAL_CY',
                'HYPERION_CODE', 'UNSPSC', 'COUNTRY_OF_ORIGIN',
                'COST_SAVINGS_LOCAL_CY', 'BASELINE_AMOUNT_LOCAL_CY',
                'SPEND_LOCAL_CY', 'PO_TERMS', 'PART_SEGMENT_CODE',
                'PULL_CANDIDATE_Y_N', 'PART_REVISION', 'UNIT_OF_MEASURE',
                'SQP_Y_N', 'CONTRACT_UNIT_PRICE_LOCAL_CY',
                'BASELINE_UNIT_PRICE_LOCAL_CY', 'ACTUAL_UNIT_PRICE_LOCAL_CY']

            return drop_col, group_by, agg_dict

        fe_logger.info("Feature engine starts to process receipts data ...\n")
        # fe_logger.info("Applying filters 'SPEND_TYPE' = 'DIRECT', "
        #                "and 'INTER_COMPANY_Y_N' = 'N' ...  \n")
        # cond1 = self.receipts['SPEND_TYPE'].str.upper() == 'DIRECT'
        # cond2 = self.receipts['INTER_COMPANY_Y_N'].str.upper() == 'N'
        cond3 = self.receipts['SBG_ID'].str.upper() == 'AER'

        self.receipts = self.receipts.loc[
            # (cond1) &
            # (cond2) &
            (cond3)]

        drop_col, group_by, agg_dict = get_trans_settings()

        self.receipts.drop(columns=drop_col, inplace=True)

        fe_logger.info("For each order, get date of {} weeks before "
                       "requested delivery date ... \n".format(BLACKOUT_WEEKS))

        self.receipts = ut.unified_dtype_conversion(self.receipts,
            ['SCHEDULED_DELIVERY_DATE', 'REQUESTED_DELIVERY_DATE',
             'ACTUAL_DELIVERY_DATE', 'CID_LAST_CODED_DATE', 'PO_PLACE_DATE',
             'BUYHON_LAST_MOD_DATE'], 'datetime')

        blackout_period = pd.Timedelta(weeks = BLACKOUT_WEEKS)
        self.receipts['BLACKOUT_DATE'] =\
            self.receipts['REQUESTED_DELIVERY_DATE'] - blackout_period

        fe_logger.info("Calculating delta in days between actual delivery "
                       "date and requested delivery date ... \n")

        self.receipts['DELTA_DAYS_ACTL_REQ_DEL_DATE'] =\
            (self.receipts['ACTUAL_DELIVERY_DATE'] -
             self.receipts['REQUESTED_DELIVERY_DATE']).dt.days

        fe_logger.info("Calculating delta in days between PO Place "
                       "date and requested delivery date ... \n")

        self.receipts['PLANNED_LEAD_DAYS'] =\
            (self.receipts['REQUESTED_DELIVERY_DATE'] -
             self.receipts['PO_PLACE_DATE']).dt.days

        fe_logger.info("Calculating delta between ordered and received "
                       "quantities ... \n")

        self.receipts['DELTA_QTY_ORDERED_RECEIVED'] = \
            (self.receipts['ORDERED_QTY'] - self.receipts['RECEIVED_QTY'])

        self.receipts = self.receipts.groupby(
            by=group_by, as_index=False).agg(agg_dict)

        fe_logger.info("Performing category columns conversion ... \n")
        cat_col = [
            'PO_NUMBER', 'PO_LINE_NUMBER', 'PART_NUMBER', 'Plant',
            'SITE_LOCAL_SUPPLIER_NUMBER', 'SUPPLIER_ID', 'SUPPLIER_NAME',
            'SITE_LOCAL_SUPPLIER_NAME', 'SCO', 'CO_CST', 'CID', 'DEMAND_PFEP',
            'AER_COE_SUPPLIER', 'SUPPLIER_UNSPSC_CODE', 'SUPPLIER_UNSPSC_DESC',
            'SUPPLIER_GEOGRAPHY', 'SUPPLIER_COUNTRY', 'SUPPLIER_GROUP_ID',
            'SUPPLIER_GROUP_NAME', 'PO_LINE_DESCRIPTION', 'MANUF_SUPP_NAME',
            'MANUF_SUPP_PART', 'PRODUCT_LINE', 'ELECTRONIC_PURCHASE',
            'BUYHON_CLASSIFICATION'
        ]
        self.receipts = ut.unified_dtype_conversion(
            self.receipts, cat_col, 'category')

        self.receipts.reset_index(drop=True, inplace=True)
        fe_logger.info("Receipts data cleansing completed ...\n")

        return self

    def _process_decommit(self):
        """process decommit and pd_commit, these two data sets have same
        column schema.

        :return: data frame that contains first decommit for each unique
        combination of PO number, PO item, PO Schedulelinenumber.
        """

        def taylor_po_change(po_change, selected_col, change_type):
            """extract information from po change history data to perform
            decommit missing information imputation

            :param po_change: po change history dataframe
            :param selected_col: selected columns in po change history
            :param change_type: string, representing ShortText in po change
            :return: data frame contains target information
            """
            for dt_col in ['PONumber', 'POItemNo', 'Schedulelinenumber',
                           'ChangeDateTime', 'NewValue', 'OldValue']:
                if dt_col not in selected_col:
                    selected_col.append(dt_col)

            po_change_partial = po_change.loc[
                po_change['ShortText'] == change_type, selected_col]

            po_change_partial['ChangeDate'] = po_change_partial[
                'ChangeDateTime'].dt.normalize()
            po_change_partial = ut.unified_dtype_conversion(
                po_change_partial, ['NewValue', 'OldValue'], 'datetime')

            po_change_partial.sort_values(
                by=['PONumber', 'POItemNo', 'Schedulelinenumber',
                    'ChangeDateTime'],
                inplace=True)

            po_change_partial.drop(columns=['ChangeDateTime'], inplace=True)
            # keep most recent information, if same change happened
            # multiple times, only save most recent record
            dedup_col = ['PONumber', 'POItemNo', 'Schedulelinenumber',
                         'OldValue']
            po_change_partial.drop_duplicates(
                subset=dedup_col, keep='last', inplace=True)

            return po_change_partial

        def impute_missing_value(decommit_all, po_change, left_on, right_on):
            """use transactional po change data to impute missing values
            in decommit data. This function can be further generalized, for now
            we only impute NewCommitDate and Date in decommit data.

            :param decommit_all: combined decommit and pd commit data
            :param po_change: po change data extracted by taylor_po_change
            :param left_on: column list in left table(decommit)
            :param right_on: column list in right table(po_change)
            :return: decommit data with miss valued imputed by po_change
            """
            decommit_all_merged = decommit_all.merge(
                right=po_change, how='left',
                left_on=left_on, right_on=right_on)

            fillna_dict = {
                'NewCommitDate': decommit_all_merged['NewValue'],
                'Date': decommit_all_merged['ChangeDate']
            }
            decommit_all_merged.fillna(value=fillna_dict, inplace=True)

            decommit_all_merged = ut.unified_dtype_conversion(
                decommit_all_merged, ['Date', 'NewCommitDate'], 'datetime')

            drop_col = po_change.columns.values.tolist()
            if 'Schedulelinenumber' in drop_col:
                drop_col.remove('Schedulelinenumber')

            decommit_all_merged.drop(columns=drop_col, inplace=True)
            decommit_all_merged.drop_duplicates(inplace=True)
            decommit_all_merged.reset_index(drop=True, inplace=True)

            return decommit_all_merged

        def pre_process_decommit(decommit_all):
            """pre process decommit data,

            NOTE: THIS FUNCTION NEEDS TO BE REDONE, FOCUS ON DATETIME
            CONVERSION.

            :param decommit_all: combined decommit and pd commit data
            :return: missing valued imputed, some other features included
            data frame
            """
            fe_logger.info(
                "For each group of PO Number/Item/Schedulelinenumber, "
                "sort by date ...\n")

            decommit_all = decommit_all.groupby(by=group_by).apply(
                lambda x: x.sort_values(by=['Date'])).reset_index(drop=True)

            decommit_all['IS_DECOMMIT'] = 1
            decommit_all = ut.unified_dtype_conversion(
                decommit_all, ['Date'], 'datetime')

            drop_col = ['ShortText', 'Vendor', 'VendorName',
                        'PurchasingGroup', 'NotificationofChange',
                        'MagnitudeofChange']
            decommit_all.drop(columns=drop_col, inplace=True)
            decommit_all.drop_duplicates(inplace=True)

            fe_logger.info(
                "Imputing missing data from PO Change History ...\n")

            selected_col = ['PONumber', 'POItemNo', 'Schedulelinenumber',
                            'NewValue', 'OldValue', 'ChangeDateTime']

            po_change_del_date = taylor_po_change(
                self.po_change, selected_col, 'ItemDeliveryDate')

            left_on = ["PONum", "POItem", "Schedulelinenumber",
                       "OriginalCommitDate"]
            right_on = ["PONumber", "POItemNo", "Schedulelinenumber",
                        "OldValue"]

            po_change_del_date = ut.unified_dtype_conversion(
                po_change_del_date, ['OldValue'], 'datetime')

            decommit_imputed_po_del_date = impute_missing_value(
                decommit_all, po_change_del_date, left_on, right_on)

            po_change_stat_date = taylor_po_change(
                self.po_change, selected_col,
                'Statistics-RelevantDeliveryDate')

            po_change_stat_date = ut.unified_dtype_conversion(
                po_change_stat_date, ['OldValue'], 'datetime')

            decommit_all_imputed = impute_missing_value(
                decommit_imputed_po_del_date, po_change_stat_date,
                left_on, right_on)

            decommit_all_imputed = ut.unified_dtype_conversion(
                decommit_all_imputed, ['NewCommitDate','OriginalCommitDate'],
                'datetime')

            return decommit_all_imputed


        group_by = [
            'PartNumber', 'Plant', 'PONum', 'POItem', 'Schedulelinenumber']

        fe_logger.info("Start to process decommits data ...\n")

        fe_logger.info("Appending decommits and PD commits ...\n")

        decommit_all = self.decommit.append(self.pd_commit)

        fe_logger.info(
            "There are {} unique PO Number/Item/Schedulelinenumber "
            "concatenation that decommitted at least once. \n".format(
                decommit_all.Concat.nunique()))

        decommit_all_proc = pre_process_decommit(decommit_all)

        fe_logger.info("Calculating previous decommits count ...\n")

        decommit_all_proc['CUMULATIVE_DECOMMIT_COUNT'] = \
            decommit_all_proc.groupby(
                by=group_by)['IS_DECOMMIT'].transform(lambda x: x.cumsum())

        decommit_all_proc['CUMULATIVE_DECOMMIT_COUNT'] = \
            decommit_all_proc['CUMULATIVE_DECOMMIT_COUNT'] - 1

        # re-compute MOC on imputed decommit data
        decommit_all_proc['MOC'] = (
                decommit_all_proc['NewCommitDate'] -
                decommit_all_proc['OriginalCommitDate']).dt.days

        fe_logger.info("Calculating previous total MOC ... \n")
        decommit_all_proc['CUMULATIVE_TOTAL_DECOMMIT_MOC'] = \
            decommit_all_proc.groupby(
            by=group_by)['MOC'].apply(lambda x: x.shift().cumsum().fillna(0))

        fe_logger.info("Converting notice of change into weeks ...\n")
        decommit_all_proc['DECOMMIT_NOC_WEEK_NUMBER'] = \
            decommit_all_proc['NOC'].apply(
                lambda x: ut.bin_notice_of_change(x))

        # need improvement, convert boolean columns to binary, convert
        # categorical column dtype to category
        decommit_all_proc = ut.map_boolean_binary(
            decommit_all_proc, 'ShorttoLead', "Yes")
        decommit_all_proc = ut.map_boolean_binary(
            decommit_all_proc, 'Contract', "Y")

        cat_col = ['SCOE', 'AEROCOE', 'PartNumber', 'Plant', 'PFEP',
                   'IDO/NONIDO', 'Concat']
        decommit_all_proc = ut.unified_dtype_conversion(
            decommit_all_proc, cat_col, 'category')

        self.decommit_proc = decommit_all_proc

        fe_logger.info("Feature engine finished processing decommit data.\n")
        return self

    def _process_po_change(self):
        """process PO change data. first use po change data as left table,
        processed decommit data as right table, and perform left join,
        on po number po item number schedulelinenumber, and

        :return: data frame by flattening po change data
        """

        def extract_po_change_by_item(change_item_str):
            """extract data from po change history according to 'ShortText'

            :param change_item_str: item change type string, from 'ShortText'
            :return: po change history data only contains specified change
            items
            """
            valid_change_list = [
                'STATISTICS-RELEVANTDELIVERYDATE', 'ITEMDELIVERYDATE',
                'ORIGINALREQUESTDATE', 'SCHEDULEDQUANTITY',
                'DELETIONINDICATORINPURCHASINGDOCUMENT',
                'PURCHASEORDERQUANTITY',
                'NETPRICEINPURCHASINGDOCUMENT(INDOCUMENTCURRENCY)',
                'SHIPPINGPOINT/RECEIVINGPOINT', 'DELIVERYPRIORITY']

            datetype_change_list = [
                'STATISTICS-RELEVANTDELIVERYDATE', 'ITEMDELIVERYDATE',
                'ORIGINALREQUESTDATE']

            if change_item_str.strip().upper() in valid_change_list:
                po_change_type = self.po_change.loc[
                    self.po_change['ShortText'] == change_item_str].copy()
                po_change_type.drop(columns=['ShortText'], inplace=True)
                if change_item_str.strip().upper() in datetype_change_list:
                    return ut.unified_dtype_conversion(
                        po_change_type, ['NewValue', 'OldValue'], 'datetime')
                return po_change_type
            else:
                raise ValueError("Invalid input item change type.")

        def process_delivery_date_change():
            """This function only transforms 'ITEMDELIVERYDATE' change in PO
            Change History data.

            :return: a new data frame with new created features.
            """
            po_single_change = extract_po_change_by_item('ITEMDELIVERYDATE')

            po_single_change.loc[:, 'DATE_CHANGE_HAPPEN_AT'] = \
                po_single_change['ChangeDateTime'].dt.normalize()

            fe_logger.info("Calculating previous delivery date change count "
                           "and magnitude ...\n")
            po_single_change.loc[:, 'MAGNITUDE_DELIVERY_DATE_CHANGE'] = \
                (po_single_change['NewValue'] -
                 po_single_change['OldValue']).dt.days

            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_DEL_DATE_CHANGE_MOC'] = \
                po_single_change.groupby(
                    by=group_by)['MAGNITUDE_DELIVERY_DATE_CHANGE'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            po_single_change.loc[:, 'CUMULATIVE_DEL_DATE_CHANGE_COUNT'] = \
                po_single_change.groupby(by=group_by).cumcount()

            fe_logger.info("Calculating previous count of delivery date "
                           "change made by NON-Honeywell employee ... \n ")

            po_single_change.loc[:,
            'CUMULATIVE_DEL_DATE_CHANGE_NON_HON_COUNT'] = \
                po_single_change.groupby(by=group_by)[
                    'IS_CHANGE_BY_EXTERNAL'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            drop_col = ['NewValue', 'ChangeDateTime', 'IS_CHANGE_BY_EXTERNAL',
                        'MAGNITUDE_DELIVERY_DATE_CHANGE']
            po_single_change.drop(columns=drop_col, inplace=True)
            po_single_change.drop_duplicates(inplace=True)
            return po_single_change

        def process_stat_del_date_change():
            """This function only transforms
            'STATISTICS-RELEVANTDELIVERYDATE' change in PO change history data

            :return: a new data frame with new created features
            """
            po_single_change = extract_po_change_by_item(
                'STATISTICS-RELEVANTDELIVERYDATE')

            po_single_change.loc[:, 'DATE_CHANGE_HAPPEN_AT'] = \
                po_single_change['ChangeDateTime'].dt.normalize()

            fe_logger.info("Calculating previous stat delivery date change "
                           "count and magnitude ...\n")
            po_single_change.loc[:, 'MAGNITUDE_STAT_DEL_DATE_CHANGE'] = \
                (po_single_change['NewValue'] -
                 po_single_change['OldValue']).dt.days

            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_STAT_DEL_DATE_CHANGE_MOC'] = \
                po_single_change.groupby(
                    by=group_by)['MAGNITUDE_STAT_DEL_DATE_CHANGE'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            po_single_change.loc[:, 'CUMULATIVE_STAT_DEL_DATE_CHANGE_COUNT']\
                = \
                po_single_change.groupby(by=group_by).cumcount()

            fe_logger.info("Calculating previous count of stat delivery date "
                           "change made by NON-Honeywell employee ... \n ")

            po_single_change.loc[
            :, 'CUMULATIVE_STAT_DEL_DATE_CHANGE_NON_HON_COUNT'] = \
                po_single_change.groupby(
                    by=group_by)['IS_CHANGE_BY_EXTERNAL'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            drop_col = ['NewValue', 'ChangeDateTime', 'IS_CHANGE_BY_EXTERNAL',
                        'MAGNITUDE_STAT_DEL_DATE_CHANGE']
            po_single_change.drop(columns=drop_col, inplace=True)
            po_single_change.drop_duplicates(inplace=True)
            return po_single_change

        def process_original_req_date():
            """This function only transforms 'ORIGINALREQUESTDATE' in po
            change history data

            :return: new data frame with new created features.
            """
            po_single_change = extract_po_change_by_item(
                'ORIGINALREQUESTDATE')

            po_single_change.loc[:, 'DATE_CHANGE_HAPPEN_AT'] = \
                po_single_change['ChangeDateTime'].dt.date

            fe_logger.info(
                "Calculating previous original request date count ...\n")
            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_ORIG_REQ_DATE_COUNT'] = \
                po_single_change.groupby(by=group_by).cumcount()

            fe_logger.info("Calculating previous count of request date "
                           "change made by NON-Honeywell employee ... \n ")
            po_single_change.loc[
            :, 'CUMULATIVE_ORIG_REQ_DATE_NON_HON_COUNT'] = \
                po_single_change.groupby(
                    by=group_by)['IS_CHANGE_BY_EXTERNAL'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            drop_col = ['NewValue', 'ChangeDateTime', 'IS_CHANGE_BY_EXTERNAL']
            po_single_change.drop(columns=drop_col, inplace=True)
            po_single_change.drop_duplicates(inplace=True)
            return po_single_change

        def process_schedule_quant():
            """This function only transforms po change on "SCHEDULEDQUANTITY"

            :return: data frame with new created features
            """
            po_single_change = extract_po_change_by_item(
                'SCHEDULEDQUANTITY')

            po_single_change.loc[:, 'DATE_CHANGE_HAPPEN_AT'] = \
                po_single_change['ChangeDateTime'].dt.date

            po_single_change = ut.convert_to_int(
                po_single_change, ['NewValue', 'OldValue'])

            fe_logger.info(
                "Calculating previous scheduled quantity change count ...\n")
            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_SCHE_QUANT_CHANGE_COUNT'] = \
                po_single_change.groupby(by=group_by).cumcount()

            fe_logger.info(
                "Calculating previous scheduled quantity change amount ...\n")
            po_single_change.loc[:, 'MAGNITUDE_SCHE_QUANT_CHANGE'] = \
                (po_single_change['NewValue'] -
                 po_single_change['OldValue'])
            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_TOTAL_SCHE_QUANT_CHANGE'] = \
                po_single_change.groupby(
                    by=group_by)['MAGNITUDE_SCHE_QUANT_CHANGE'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            fe_logger.info("Calculating previous scheduled quantity "
                           "change made by NON-Honeywell employee ... \n ")

            po_single_change.loc[
            :, 'CUMULATIVE_SCHE_QUANT_CHANGE_NON_HON_COUNT'] = \
                po_single_change.groupby(
                    by=group_by)['IS_CHANGE_BY_EXTERNAL'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            drop_col = ['NewValue', 'OldValue', 'ChangeDateTime',
                        'IS_CHANGE_BY_EXTERNAL',
                        'MAGNITUDE_SCHE_QUANT_CHANGE']
            po_single_change.drop(columns=drop_col, inplace=True)
            po_single_change.drop_duplicates(inplace=True)
            return po_single_change

        def process_po_quant():
            """This function only transforms po change on
            "PURCHASEORDERQUANTITY "

            :return: data frame with new created features
            """
            po_single_change = extract_po_change_by_item(
                'PURCHASEORDERQUANTITY')

            po_single_change.loc[:, 'DATE_CHANGE_HAPPEN_AT'] = \
                po_single_change['ChangeDateTime'].dt.date

            po_single_change = ut.convert_to_int(
                po_single_change, ['NewValue', 'OldValue'])

            fe_logger.info(
                "Calculating previous PO quantity change count ...\n")
            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_PO_QUANT_CHANGE_COUNT'] = \
                po_single_change.groupby(by=group_by).cumcount()

            fe_logger.info(
                "Calculating previous PO quantity change amount ...\n")
            po_single_change.loc[:, 'MAGNITUDE_PO_QUANT_CHANGE'] = \
                (po_single_change['NewValue'] -
                 po_single_change['OldValue'])
            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_TOTAL_PO_QUANT_CHANGE'] = \
                po_single_change.groupby(
                    by=group_by)['MAGNITUDE_PO_QUANT_CHANGE'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            fe_logger.info("Calculating previous PO quantity "
                           "change made by NON-Honeywell employee ... \n ")

            po_single_change.loc[
            :, 'CUMULATIVE_PO_QUANT_CHANGE_NON_HON_COUNT'] = \
                po_single_change.groupby(
                    by=group_by)['IS_CHANGE_BY_EXTERNAL'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            drop_col = ['NewValue', 'OldValue', 'ChangeDateTime',
                        'IS_CHANGE_BY_EXTERNAL',
                        'MAGNITUDE_PO_QUANT_CHANGE']
            po_single_change.drop(columns=drop_col, inplace=True)
            po_single_change.drop_duplicates(inplace=True)
            return po_single_change

        def process_del_po_doc():
            """This function only transforms po change on
            "DELETIONINDICATORINPURCHASINGDOCUMENT"

            :return: data frame with new created features
            """
            po_single_change = extract_po_change_by_item(
                'DELETIONINDICATORINPURCHASINGDOCUMENT')

            po_single_change.loc[:, 'DATE_CHANGE_HAPPEN_AT'] = \
                po_single_change['ChangeDateTime'].dt.date

            fe_logger.info(
                "Calculating previous delete PO document change count ...\n")
            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:, 'CUMULATIVE_DELETE_PO_DOC_CHANGE_COUNT']\
                = \
                po_single_change.groupby(by=group_by).cumcount()

            fe_logger.info("Calculating previous PO quantity "
                           "change made by NON-Honeywell employee ... \n ")

            po_single_change.loc[
            :, 'CUMULATIVE_DELETE_PO_DOC_CHANGE_NON_HON_COUNT'] = \
                po_single_change.groupby(
                    by=group_by)['IS_CHANGE_BY_EXTERNAL'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            drop_col = ['NewValue', 'OldValue', 'ChangeDateTime',
                        'IS_CHANGE_BY_EXTERNAL']
            po_single_change.drop(columns=drop_col, inplace=True)
            po_single_change.drop_duplicates(inplace=True)
            return po_single_change

        def process_netprice_po_doc():
            """This function only transforms po change on
            "NETPRICEINPURCHASINGDOCUMENT(INDOCUMENTCURRENCY)"

            :return: data frame with new created features
            """
            po_single_change = extract_po_change_by_item(
                'NETPRICEINPURCHASINGDOCUMENT(INDOCUMENTCURRENCY)')

            po_single_change.loc[:, 'DATE_CHANGE_HAPPEN_AT'] = \
                po_single_change['ChangeDateTime'].dt.date

            fe_logger.info(
                "Calculating previous net price purchase document change "
                "count ...\n")
            group_by = ['PONumber', 'POItemNo', 'Schedulelinenumber']
            po_single_change.loc[:,
            'CUMULATIVE_NETPRICE_PO_DOC_CHANGE_COUNT'] = \
                po_single_change.groupby(by=group_by).cumcount()

            fe_logger.info("Calculating previous net price PO document "
                           "change made by NON-Honeywell employee ... \n ")

            po_single_change.loc[
            :, 'CUMULATIVE_NETPRICE_PO_DOC_CHANGE_NON_HON_COUNT'] = \
                po_single_change.groupby(
                    by=group_by)['IS_CHANGE_BY_EXTERNAL'].apply(
                    lambda x: x.shift().cumsum().fillna(0))

            drop_col = ['NewValue', 'OldValue', 'ChangeDateTime',
                        'IS_CHANGE_BY_EXTERNAL']
            po_single_change.drop(columns=drop_col, inplace=True)
            po_single_change.drop_duplicates(inplace=True)
            return po_single_change

        fe_logger.info("Processing PO Change History data ...\n")

        self.po_change['ShortText'] = \
            self.po_change['ShortText'].str.upper().str.strip()

        # note here we do not include Schedulelinenumber because when change
        # item (ShortText) is "OriginalRequestDate",
        # then 'Schedulelinenumber' can be blank.
        group_by_list = ['PONumber', 'POItemNo', 'ShortText']

        self.po_change = self.po_change.groupby(by=group_by_list).apply(
            lambda x: x.sort_values(
                by=['ChangeDateTime'])).reset_index(drop=True)

        fe_logger.info("Labeling change item if the change was made by "
                       "NON-Honeywell employee ....\n")
        self.po_change.loc[:, 'IS_CHANGE_BY_EXTERNAL'] = \
            self.po_change['UserEID'].apply(
                lambda x: 0 if ut.valid_honeywell_eid(x) else 1)

        drop_col = ['Header_ItemChange', 'UserEID', 'Versionnumber']
        self.po_change.drop(columns=drop_col, inplace=True)

        fe_logger.info("Flattening PO Change History data ... \n")
        proc_del_date = process_delivery_date_change()
        proc_stat_date = process_stat_del_date_change()
        proc_req_date = process_original_req_date()
        proc_sche_quant = process_schedule_quant()
        proc_po_quant = process_po_quant()
        proc_del_po_doc = process_del_po_doc()
        proc_netprice_po_doc = process_netprice_po_doc()

        self.po_change_flattened = proc_del_date.append(
            [proc_stat_date, proc_req_date, proc_sche_quant,
             proc_po_quant, proc_del_po_doc, proc_netprice_po_doc],
            sort=False)
        sort_by = ['PONumber', 'POItemNo', 'Schedulelinenumber',
                   'DATE_CHANGE_HAPPEN_AT']
        self.po_change_flattened.sort_values(by=sort_by, inplace=True)

        fe_logger.info("Filling missing values in flattened po change "
                       "data ...\n")

        self.po_change_flattened = self.po_change_flattened.groupby(
            by=['PONumber', 'POItemNo', 'Schedulelinenumber'],
            as_index=False).fillna(method='ffill')
        self.po_change_flattened.fillna(0, inplace=True)
        self.po_change_flattened = ut.convert_to_datetime(
            self.po_change_flattened, ['OldValue', 'DATE_CHANGE_HAPPEN_AT'])

        # fe_logger.info("Joining PO change and receipts data for requested "
        #                "delivery date at PO Item level ... \n")
        #
        #
        # right_on = ['PO_NUMBER', 'PO_LINE_NUMBER', 'PART_NUMBER']
        # selected_col = right_on + ['REQUESTED_DELIVERY_DATE']
        # left_on = ['PONumber', "POItemNo", "PartNumber"]
        # kwargs = {
        #     'left': self.po_change_flattened,
        #     'right': self.receipts.loc[:, selected_col],
        #     'left_on': left_on,
        #     'right_on': right_on,
        #     'how': 'left'
        # }
        # self.po_change_flattened = ut.merge_and_drop(
        #     [x for x in right_on if x != 'Plant'], **kwargs)

        # fe_logger.info("Caculating NOC week number using requested "
        #                "deliverydate from receipt data as time horizon ... \n")
        #
        # self.po_change_flattened.dropna(subset=['REQUESTED_DELIVERY_DATE'],
        #                                 inplace=True)
        # self.po_change_flattened['CHANGE_HAPPEN_AT_WEEK'] =\
        #     np.floor(
        #         (self.po_change_flattened['DATE_CHANGE_HAPPEN_AT'] -
        #          self.po_change_flattened[
        #              'REQUESTED_DELIVERY_DATE'])/np.timedelta64(1,'W')).astype(
        #         'int')


        self.po_change_flattened = self.po_change_flattened[
            self.po_change_flattened['Schedulelinenumber'] != 'nan']
        self.po_change_flattened.drop_duplicates(inplace=True)

        fe_logger.info("Feature engine finished processing PO Change "
                       "History data.\n")
        return self

    def _process_material_master(self):
        """process material master table (mm)

        :return:
        """
        fe_logger.info("Start material master data cleansing ...\n")

        drop_col = ['Scheduled Receipts Issue Date', 'Date',
                    'Product Hierarchy', 'Valuation Class',
                    'Part Family', 'Production Scheduler', 'Demand Type']

        fe_logger.info("Removing redundant or all missing columns ...\n")
        self.mm.drop(columns=drop_col, inplace=True)
        self.mm.drop_duplicates(inplace=True)

        fe_logger.info("Converting columns as categorical ... \n")

        cat_col = [
            'Part Number', 'Site', 'PFEP', 'Supply Type', 'PFEP Supply',
            'Valuation Type', 'ABC Code', 'Procure Method', 'Planner Code'
        ]
        self.mm = ut.unified_dtype_conversion(self.mm, cat_col, 'category')

        fe_logger.info("Materials master data cleansing completed ...\n")
        return self

    def __merge_base_receipts(self):
        """ this function handles merging base with receipts data.

        :param base: base dataframe in feature engine instance
        :param receipts: receipts dataframe in feature engine instance
        :return: merged dataframe
        """
        left_on = ['PONumber', 'POItemNo', 'PartNumber', 'Plant', 'vendor2']
        right_on = ['PO_NUMBER', 'PO_LINE_NUMBER', 'PART_NUMBER', 'Plant',
                    'SITE_LOCAL_SUPPLIER_NUMBER']

        selected_receipts_col = right_on +\
            ["SCHEDULED_DELIVERY_DATE", "REQUESTED_DELIVERY_DATE",
             "ACTUAL_DELIVERY_DATE", "ORDERED_QTY", "RECEIVED_QTY",
             "SITE_LEAD_TIME", "PO_NET_DAYS", "ACTUAL_UNIT_PRICE",
             "CONTRACT_UNIT_PRICE", "SPEND", "PO_PLACE_DATE",
             "ACTUAL_LEAD_TIME", "BUYHON_CONFIDENCE_LEVEL", "BLACKOUT_DATE",
             "DELTA_DAYS_ACTL_REQ_DEL_DATE", "PLANNED_LEAD_DAYS",
             "DELTA_QTY_ORDERED_RECEIVED"]

        kwargs = {
            'left': self.base,
            'right': self.receipts[selected_receipts_col],
            'left_on': left_on,
            'right_on': right_on,
            'how': 'left'
        }

        drop_col = ['PO_NUMBER', 'PO_LINE_NUMBER', 'PART_NUMBER',
                    'SITE_LOCAL_SUPPLIER_NUMBER']

        return ut.merge_and_drop(drop_col, **kwargs)


    def _put_all_parts_together(self):
        """construct a training data set

        :return: data frame
        """
        fe_logger.info("Feature engine is performing data fusion ... \n")

        merged = self.__merge_base_receipts()

        fe_logger.info(
            "Adding processed decommits to transformed PO change data ... \n")

        selected_col = [
            'PONum', 'POItem', 'Schedulelinenumber', 'OriginalCommitDate',
            'Date', 'IS_DECOMMIT', 'DECOMMIT_NOC_WEEK_NUMBER']
        left_on = [
            'PONumber', 'POItemNo', 'Schedulelinenumber', 'OldValue',
            'DATE_CHANGE_HAPPEN_AT']
        right_on = [
            'PONum', 'POItem', 'Schedulelinenumber',
            'OriginalCommitDate', 'Date']
        decommit = self.decommit_proc.loc[:, selected_col]

        drop_col = [
            'PONum', 'POItem', 'OriginalCommitDate',
            'DATE_CHANGE_HAPPEN_AT', 'Date', 'OldValue']

        kwargs = {
            'left': self.po_change_flattened,
            'right': decommit,
            'left_on': left_on,
            'right_on': right_on,
            'how': 'left'
        }

        po_change_decommit = ut.merge_and_drop(
            drop_col=drop_col, **kwargs)

        po_change_decommit.drop_duplicates(inplace=True)
        po_change_decommit.reset_index(drop=True, inplace=True)

        fe_logger.info("Merging new features with base data set ... \n")

        left_on = ['PONumber', 'POItemNo', 'Schedulelinenumber',
                   'Vendor', 'PurchasingGroup', 'PartNumber', 'Plant',
                   'VendorName', 'vendor2']
        right_on = left_on

        kwargs = {
            'left': self.base,
            'right': po_change_decommit,
            'left_on': left_on,
            'right_on': right_on,
            'how': 'left'
        }

        self.final = ut.merge_and_drop(drop_col=[], **kwargs)

        self.final['DECOMMIT_NOC_WEEK_NUMBER'].fillna(
            NO_DECOMMIT, inplace=True)

        self.final['IS_DECOMMIT'].fillna(0, inplace=True)

        self.final.reset_index(inplace=True, drop=True)

        return self


    def _save_to_file(self, format='csv', path=PROC_DATA_PATH):
        """save data to pickle/CSV file under PROC_DATA_PATH

        :param format: file format to save data
        :param path: directory to save the data, by default save file under
        PROC_DATA_PATH
        :return:
        """
        fe_logger.info(
            "Saving processed data set at {}. \n".format(path))

        file_format = format.strip().lower()

        file_dict = {
            "po_change_flattened": self.po_change_flattened,
            "decommit_proc": self.decommit_proc,
            "final": self.final
        }

        for key, value in file_dict.items():
            file_name = key + ".{}".format(file_format)

            if file_format == 'csv':
                value.to_csv(os.path.join(path, file_name), index=False)

            elif file_format == 'pickle':
                value.to_pickle(os.path.join(path, file_name))
            else:
                raise ValueError("{} is not supported yet.\n".format(format))


    def run(self):
        """entry point for feature engine

        :return: data frame that is ready for model training.
        """
        self._process_base()
        self._process_receipts()
        self._process_decommit()
        self._process_material_master()
        self._process_po_change()
        self._put_all_parts_together()
        self._save_to_file(format='csv')
        fe_logger.info("Feature engine completed all tasks.\n")
