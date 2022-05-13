import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle

def null_values(data_path):
    """
    calculate and save the fraction of null values
    :param data_path: data path
    :return: None
    """
    data = pd.read_csv(data_path, delimiter='|')
    N = data.shape
    print(f"data shape:   {N} ")
    # calc the fraction of the Null values in each column
    print("data  NaN")
    df_nan = data.isna().sum()
    df_nan = pd.DataFrame({'column': df_nan.index, 'count_NaN': df_nan.values})
    print("--------------")
    df_nan['frac'] = df_nan['count_NaN'].apply(lambda x: x/N[0])
    df_nan.to_csv('NaN_Data.csv')

    # calc the fraction of patients which have Null values in all column - in each column
    data_path = "full_df_train.psv"
    data = pd.read_csv(data_path, delimiter='|')
    df = data.groupby('patient').apply(lambda x: x.isnull().all()).sum()
    print(f"df shape:   {df.shape}")
    df = pd.DataFrame({'column': df.index, 'count_NaN': df.values})
    df['frac'] = df['count_NaN'].apply(lambda x: x/data.shape[0])
    df.to_csv("NaNData_2.csv")


def corr_test(data, col1, col2):
    """
     conduct correlation test between col1 and col2
    :param data: dataset - pandas
    :param col1: column 1 name
    :param col2: column 2 name
    :return: (correlation coefficient, p-value)
    """
    data = data[[col1, col2]]
    data = data.fillna(0)
    l1 = data[col1].tolist()
    l2 = data[col2].tolist()
    return stats.pearsonr(l1, l2)


def t_test(data, col):
    """
    conduct T-test:
     null hypothesis: the col values with label=0 and the col values with label=1 have identical average
    and print the statistic and p-value
    :param data: dataframe - pandas
    :param col: column name
    :return:None
    """
    l1 = data[data['label'] == 0][col].tolist()
    l2 = data[data['label'] == 1][col].tolist()
    res = stats.ttest_ind(l1, l2, equal_var=False)
    print("T-test   col name = ", col)
    print(f"stat: {res[0]},  p-value: {res[1]}")


if __name__ == '__main__':
    data_path = "full_df_mean_train.psv"
    data = pd.read_csv(data_path, delimiter='|')

    # calc null values
    # null_values(data_path) # TODO remove #

    # corr test
    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
                'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Magnesium',
               'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets', 'Age',
               'HospAdmTime', 'ICULOS']

    done_list = []
    cols_list = []
    c = 0
    for c1 in columns:
        for c2 in columns:
            if c1 != c2 and (c1, c2) not in done_list and (c2, c1) not in done_list:
                done_list.append((c1, c2))
                res = corr_test(data, c1, c2)
                if res[0] > 0.5 or res[0] < -0.5:
                    c += 1
                    cols_list.append((c1,c2))
                    print(f"col1 = {c1} , col2 = {c2}")
                    print(f"Pearson’s correlation coefficient: {res[0]} \n p-value: {res[1]}")
                    print("---------------------------")
    print(f"count = {c}")
    with open('cols_list_scatter.pkl', 'wb') as f:
        pickle.dump(cols_list, f)

""" # t test
    data_path = "full_df_mean_train.psv"
    data = pd.read_csv(data_path, delimiter='|')
    t_test(data, 'ICULOS')"""


