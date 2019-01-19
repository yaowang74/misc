# -*- coding: utf-8 -*-
"""
Utility functions

@author: H211803
"""
import errors as err
import pandas as pd
import numpy as np
from config import NO_DECOMMIT
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def valid_honeywell_eid(eid_str):
    """Performing honeywell EID validation. Return True if a input string is of
    Honeywell EID format, False otherwise.

    NOTE: AN EMPTY STRING WILL NOT BE CONSIDERED AS VALID HONEYWELL EID.

    :param eid_str: input string representing the user id.
    :return:
    """
    id = eid_str.strip().upper()
    if len(id) != 7:
        return False
    else:
        cond1 = id.startswith("H") or id.startswith('E')
        cond2 = id[1:].isdigit()
        cond3 = id.strip().upper() == 'ZAUTSYS'
        if (cond1 and cond2) or cond3:
            return True
        else:
            return False


def convert_to_datetime(df, col):
    """convert columns in input dataframe to datetime dtype. NOTE: this
    function will normalize the time value in datetime column.

    :param df: input dataframe that contains columns specified in col
    :param col: column name list
    :return: dataframe with specified columns changed to datetime dtype.
    """
    if isinstance(col, str):
        col_list = [col]
    elif isinstance(col, list):
        col_list = col
    else:
        raise err.InputError()

    for element in col_list:
        if element not in df.columns.values.tolist():
            raise err.InputError()

    df.loc[:, col_list] = df[col_list].apply(
        lambda x: pd.to_datetime(
            x, errors='coerce', format='%Y-%m-%d').dt.normalize())

    return df


def convert_to_int(df, col):
    """convert columns values in df to integer

    :param df: input dataframe that contains columns specified in col
    :param col: column names list
    :return: dataframe with specified columns changed to int dtype.
    """
    if isinstance(col, str):
        col_list = [col]
    elif isinstance(col, list):
        col_list = col
    else:
        raise err.InputError()

    for element in col_list:
        if element not in df.columns.values.tolist():
            raise err.InputError()
    df.loc[:, col_list] = df[col_list].apply(pd.to_numeric)
    return df


def unified_dtype_conversion(df, col, convert_to):
    """convert columns in df to category data types.

    :param df: input dataframe that contains columns specified in col
    :param col: column names list
    :param convert_to: dtype string that will be converted to
    :return: dataframe with specified columns changed to specified dtype.
    """
    if isinstance(col, str):
        col_list = [col]
    elif isinstance(col, list):
        col_list = col
    else:
        raise err.InputError()

    for element in col_list:
        if element not in df.columns.values.tolist():
            raise err.InputError()

    dtype = convert_to.strip().lower()
    if dtype == 'category':
        df.loc[:, col_list] = df[col_list].apply(
            lambda x: x.astype('category'))
    elif dtype == 'int':
        df.loc[:, col_list] = df[col_list].apply(pd.to_numeric)
    elif dtype == 'datetime':
        df.loc[:, col_list] = df[col_list].apply(
            lambda x: pd.to_datetime(
                x, errors='coerce', format='%Y-%m-%d').dt.normalize())
    else:
        raise ValueError("{} has not been added to unified conversion "
                         "function yet, please update it in utils "
                         "module.".format(dtype))
    return df


def map_boolean_binary(df, col, yes_str):
    """convert columns of boolean strings (Yes/No) to 1/0

    :param df: target dataframe
    :param col: list of columns in df that needed to be converted
    :param yes_str: string representing yes
    :param no_str: string representing no
    :return: transformed dataframe
    """
    if isinstance(col, str) and col.strip() != '':
        df[col] = pd.Series(
            np.where(df[col].values == yes_str, 1, 0), df.index)
        return df
    else:
        raise err.InputError()


def bin_notice_of_change(noc):
    """produce week number from item delivery date backward.

    :param noc: integer, representing notice of change
    :return: string labeling week number for a noc
    """
    week = 'WEEK_'
    if not isinstance(noc, int) or noc < 0:
        raise ValueError("NOC cannot be negative number.")

    for w in range(0, 14):
        if noc >= 7 * w and noc < 7 * (w + 1):
            if w == 13:
                return NO_DECOMMIT
            return week+str(w + 1)


def id_look_up(id_str, target_df, target_col):
    """this function performs simply id look up in target_df using
    target_col and user specified id_str.

    :param id_str: id string that will be looked for, assuming all id type
    of columns are loaded as string, this was done in config file.
    :param target_df: data frame that user will look up id
    :param target_col: column names that possibly contains is string in
    dataframe
    :return: data frame that only contains relevant information.
    """
    if not (id_str.strip() or target_col.strip()) :
        raise ValueError("Argument 'id_str' and 'target_col' cannot be null.")

    return target_df.loc[target_df[target_col] == id_str]


def merge_and_drop(drop_col, **kwargs):
    """this function performs dataframe merging and drop columns in merged
    data frame

    :param drop_col: list of columns that will be dropped in merged dataframe
    :param kwargs: key word arguments in pandas.merge
    :return: merged dataframe with drop_col dropped.
    """
    if kwargs['left'].empty or kwargs['right'].empty:
        raise ValueError("One of the input dataframe is empty.")

    if not isinstance(drop_col, list):
        raise ValueError("Argument 'drop_col' must be a list.")

    merged = pd.merge(
        left=kwargs['left'], right=kwargs['right'], how=kwargs['how'],
        left_on=kwargs['left_on'], right_on=kwargs['right_on'])

    return merged.drop(columns=drop_col)


def keras_to_categorical(categorical_series):
    """This function will convert categorical column which is not of 'int'
    dtype to onehot encoded representation.

    :param categorical_series: the pandas series that need to be converted
    :return: onehot encoded representation of input columns, number of
    classes and encoder for inverse_transform the prediction results
    """
    num_classes = categorical_series.nunique()
    encoder = LabelEncoder()
    encoder.fit(categorical_series)
    encoded_y = encoder.transform(categorical_series)

    encoded_y = to_categorical(encoded_y)

    return encoded_y, num_classes, encoder


def concat_df_horizontally(df_list):
    """this function simply horizontally concatenate dataframes in df_list

    :param df_list: list of dataframes. These dataframes should have same
    length.
    :return: concatenated dataframe
    """
    return pd.concat(objs=df_list, axis=1)


def h2o_to_factor(h2o_frame, cat_col_list):
    """convert columns in h2o_frame specified in cat_col_list to factor

    :param h2o_frame: h2o frame, not pandas dataframe
    :param cat_col_list: column name list that are categorical
    :return: h2o frame with categorical columns converted to factor.
    """
    for col in cat_col_list:
        h2o_frame[col] = h2o_frame[col].asfactor()

    return h2o_frame








