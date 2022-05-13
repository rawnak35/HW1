import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stage = 1
choosen_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'BaseExcess', 'FiO2', 'pH',
              'SaO2', 'AST', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Magnesium',
              'Phosphate', 'Potassium', 'Bilirubin_total', 'Hgb', 'PTT', 'WBC', 'Platelets', 'Age',
              'HospAdmTime', 'ICULOS', 'Gender', 'Unit1', 'Unit2',"BUN"]


def count_NaN(path):
    nan_ratio_dict = {}
    df = pd.read_csv(path, delimiter="|")
    """for column in full_df.columns:
        count_nan = full_df[column].isna().sum()
        """
    # Corelation between "SepsisLabel" and NaN values
    #TODO:delete columns from dataframe
    #choosen_columns =  ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel', 'label']
    choosen_columns = df.columns
    for column in choosen_columns:
        count_nan = df[column].isna().sum()
        nan_ratio = round(count_nan / len(df[column].to_list()), 2)
        nan_ratio_dict[column] = nan_ratio
    return nan_ratio_dict



def fillna_with_mean(path, delimiter):
    df = pd.read_csv(path, delimiter=delimiter)
    mean_dict1 = {}
    mean_dict0 = {}
    weighted_avg = {}
    normal_avg = {}
    df_1 = df[df['label']==1]
    df_0 = df[df['label']==0]
    for column in choosen_columns:
        tmp1 = df_1[column]
        avg1 = tmp1.mean()
        mean_dict1[column] = avg1
        tmp0 = df_0[column]
        avg0 = tmp0.mean()
        mean_dict0[column] = avg0

        avg = df[column].mean()
        normal_avg[column] = avg

        num_1 = tmp1.count()
        num_0 = tmp0.count()
        w_avg = (avg1*num_1 + avg0*num_0)/(num_0+num_1)
        weighted_avg[column] = w_avg

    return mean_dict1, mean_dict0, normal_avg, weighted_avg



train_path = "full_df_mean_train.psv"
test_path = "full_df_mean_test.psv"
if stage == 0: #NaN values
    count_NaN(train_path)

if stage == 1: # Fill NaN with avg calculated according to the label and column
    dict1, dict0, normal_avg, weighted_avg = fillna_with_mean(train_path, "|")
    """print(dict0)
    print("----------------------------------")
    print(dict1)"""
    nan_ratio_dict = count_NaN(train_path)
    df = pd.read_csv(train_path, delimiter="|")
    columns = choosen_columns.copy()
    columns.append('label')
    df = df[columns]
    m1 = df['label'] == 1
    m0 = df['label'] == 0
    for column in choosen_columns:
        if nan_ratio_dict[column] > 0.5:
            df[column] = df[column].fillna(0)
        else:
            df.loc[m1, column] = df.loc[m1, column].fillna(dict1[column])
            df.loc[m0, column] = df.loc[m0, column].fillna(dict0[column])
    df.to_csv("full_df_mean_train_fillAvg0.psv", sep="|")

    """test_df = pd.read_csv(test_path, delimiter="|")
    test_df_norm = test_df[columns].copy()
    test_df_weighted = test_df[columns].copy()
    for column in choosen_columns:
        test_df_norm[column] = test_df_norm[column].fillna(normal_avg[column])
        test_df_weighted[column] = test_df_weighted[column].fillna(weighted_avg[column])

    test_df_norm.to_csv("full_df_mean_test_fillNormAvg.psv", sep="|")
    test_df_norm.to_csv("full_df_mean_test_fillWAvg.psv", sep="|")"""

if stage == 2:
    _, _, normal_avg, weighted_avg = fillna_with_mean(train_path, "|")
    nan_ratio_dict = count_NaN(train_path)
    columns = choosen_columns.copy()
    columns.append('label')


    test_df = pd.read_csv(test_path, delimiter="|")
    test_df = test_df[columns]
    test_df_norm = test_df[columns].copy()
    test_df_weighted = test_df[columns].copy()
    for column in choosen_columns:
        if nan_ratio_dict[column] > 0.5:
            test_df_norm[column] = test_df_norm[column].fillna(0)
            test_df_weighted[column] = test_df_weighted[column].fillna(0)
        else:
            test_df_norm[column] = test_df_norm[column].fillna(normal_avg[column])
            test_df_weighted[column] = test_df_weighted[column].fillna(weighted_avg[column])

    test_df_norm.to_csv("full_df_mean_test_fillNormAvg0.psv", sep="|")
    test_df_norm.to_csv("full_df_mean_test_fillWAvg0.psv", sep="|")
