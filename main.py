import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
directory = "data/Test"
sep_directory = "data_sep/Test"


def load_data_for_model(features, directory, path_out):
    i = 0
    for filename in os.listdir(directory):
        print("==============================")
        full_path = directory + "/" + filename
        print(full_path)
        df = pd.read_csv(full_path, delimiter="|")

        # print(df.head())
        SepsisLabel = df['SepsisLabel'].to_list()
        try:
            idx = SepsisLabel.index(1)
            label = 1
        except:
            idx = len(SepsisLabel)
            label = 0
        print(idx)

        df = df.truncate(after=idx)
        patient_id = filename[8:-4]
        df = df.fillna(0)
        print()
        values = df[features].values
        df1 = pd.DataFrame({"patient": [patient_id], "values": [values], 'label': [label]})
        df1['values'] = df1['values']
        if i == 0:
            full_df = df1.__deepcopy__()
            i += 1
        else:
            full_df = pd.concat([full_df, df1])
    full_df.to_pickle(path_out)


def plot_hist(column_list, column_name):
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=column_list, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(column_name)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig("plots"+"/"+column_name)
    plt.clf()


stage = 2 #TODO
if stage == 0:#Iterate over files in directory and read each file as pandas dataframe
    i = 0
    for filename in os.listdir(directory):
        print("==============================")
        full_path = directory +"/"+filename
        print(full_path)
        df = pd.read_csv(full_path, delimiter="|")

        #print(df.head())
        SepsisLabel = df['SepsisLabel'].to_list()
        try:
            idx = SepsisLabel.index(1)
            label = 1
        except:
            idx = len(SepsisLabel)
            label = 0
        print(idx)
        df = df.truncate(after=idx)
        df['label'] = label
        df['patient'] = filename[8:-4]
        print()

        if i == 0:
            full_df = df.__deepcopy__()
            i += 1
        else:
            full_df = pd.concat([full_df, df])

        new_full_path = sep_directory + "/" + filename
        df.to_csv(new_full_path, sep="|")


    full_df.to_csv("full_df_train.psv",sep="|")
    stage = 1

if stage == 1: #NaN values
    full_df = pd.read_csv("full_df_train.psv", delimiter="|")
    """for column in full_df.columns:
        count_nan = full_df[column].isna().sum()
        print(column, "----", count_nan, "out of: ", len(full_df[column].to_list()))"""
    # Corelation between "SepsisLabel" and NaN values
    # Filter the full dataframe by SepsisLabel, keep oly rows that contane SepsisLabel = 1
    filtered_df = full_df[full_df.SepsisLabel == 1]
    chosen_columns = []
    # print(filtered_df.head())
    for column in filtered_df.columns:
        count_nan_full = full_df[column].isna().sum()
        nan_ratio_full = round(count_nan_full/len(full_df[column].to_list()),2)
        if nan_ratio_full < 0.7:
            if column != "patient":

                chosen_columns.append(column)
        count_nan_filtered = filtered_df[column].isna().sum()
        nan_ratio_filtered = round(count_nan_filtered/len(filtered_df[column].to_list()),2)
        print("----------------", column,"----------------")
        print("full DF -->", nan_ratio_full)
        print("filtered DF -->", nan_ratio_filtered)
    print("chosen columns:")
    print(chosen_columns)

    for column in chosen_columns:
        column_list = full_df[column].to_list()
        plot_hist(column_list,column)

# Iterate over files in directory and read each file as pandas dataframe and save to one df the
# mean values of each patient
if stage == 2:
    i = 0
    for filename in os.listdir(directory):
        print("==============================")
        full_path = directory + "/" + filename
        print(full_path)
        df = pd.read_csv(full_path, delimiter="|")

        # print(df.head())
        SepsisLabel = df['SepsisLabel'].to_list()
        try:
            idx = SepsisLabel.index(1)
            label = 1
        except:
            idx = len(SepsisLabel)
            label = 0
        print(idx)
        df = df.truncate(after=idx)
        df['label'] = label

        df = df.mean(axis=0).to_frame().T
        df['patient'] = filename[8:-4]
        if i == 0:
            full_df = df.__deepcopy__()
            i += 1
        else:
            full_df = pd.concat([full_df, df])
    full_df.to_csv("full_df_mean_test.psv", sep="|")


features = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
              'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Magnesium',
              'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets', 'Age',
              'HospAdmTime', 'ICULOS', 'Gender', 'Unit1', 'Unit2']
