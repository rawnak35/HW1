from sklearn.linear_model import LogisticRegression
import pandas as pd
import sklearn.model_selection as sk
from sklearn.metrics import f1_score
import pickle
from sklearn.feature_selection import RFE
from itertools import compress
from sklearn.preprocessing import StandardScaler

def option1(x_cols):
    """
    fillna = 0
    all features
    :return:
    """
    data_path = "full_df_mean_train.psv"
    data = pd.read_csv(data_path, delimiter='|')
    data = data.fillna(0)


    y_col = ['label']
    x_train, x_test, y_train, y_test = sk.train_test_split(data[x_cols], data[y_col])

    lr = LogisticRegression(max_iter=10000, penalty=None)
    lr.fit(x_train, y_train)
    y_pred_train = lr.predict(x_train)
    y_pred_test = lr.predict(x_test)
    print(f"f1 score train: {f1_score(y_train, y_pred_train)}")
    print(f"f1 score test: {f1_score(y_test, y_pred_test)}")
    print("score on test: " + str(lr.score(x_test, y_test)))
    print("score on train: " + str(lr.score(x_train, y_train)))


def option1_all_data(x_cols, norm=False):
    data_train_path = "full_df_mean_train.psv"
    train_data = pd.read_csv(data_train_path, delimiter='|')
    train_data = train_data.fillna(0)
    data_test_path = "full_df_mean_test.psv"
    test_data = pd.read_csv(data_test_path, delimiter='|')
    test_data = test_data.fillna(0)


    y_col = ['label']
    #x_train, x_test, y_train, y_test = sk.train_test_split(x_train, data[y_col])
    x_train = train_data[x_cols].values
    x_test = test_data[x_cols].values
    y_train = train_data[y_col].values.ravel()
    y_test = test_data[y_col].values.ravel()
    print(x_train.shape)
    if norm:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
    print(x_train[0])

    lr = LogisticRegression(max_iter=1000,random_state=0)
    lr.fit(x_train, y_train)

    y_pred_train = lr.predict(x_train)
    y_pred_test = lr.predict(x_test)
    print(f"f1 score train: {f1_score(y_train, y_pred_train)}")
    print(f"f1 score test: {f1_score(y_test, y_pred_test)}")
    print("score on test: " + str(lr.score(x_test, y_test)))
    print("score on train: " + str(lr.score(x_train, y_train)))


def option1_all_data_RNF(x_cols):
    data_train_path = "full_df_mean_train.psv"
    train_data = pd.read_csv(data_train_path, delimiter='|')
    train_data = train_data.fillna(0)
    data_test_path = "full_df_mean_test.psv"
    test_data = pd.read_csv(data_test_path, delimiter='|')
    test_data = test_data.fillna(0)
    y_col = ['label']
    #x_train, x_test, y_train, y_test = sk.train_test_split(x_train, data[y_col])
    x_train = train_data[x_cols]
    x_test = test_data[x_cols]
    y_train = train_data[y_col]
    y_test = test_data[y_col]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    lr = LogisticRegression(max_iter=1000)
    selector = RFE(lr, n_features_to_select=20, step=1)
    selector = selector.fit(x_train, y_train)
    selector.fit(x_train, y_train)
    mask = selector.support_
    #print(len(mask))
    featuers_list = list(compress(x_cols,mask))
    with open('model_op1_RFE.pkl', 'wb') as f:
        pickle.dump(lr, f)
    y_pred_train = selector.predict(x_train)
    y_pred_test = selector.predict(x_test)
    print(f"f1 score train: {round(f1_score(y_train, y_pred_train),3)}")
    print(f"f1 score test: {round(f1_score(y_test, y_pred_test),3)}")
    print("score on test: " + str(selector.score(x_test, y_test)))
    print("score on train: " + str(selector.score(x_train, y_train)))
    print("-------")
    print(featuers_list)
    print(len(featuers_list))

x_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
                   'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Magnesium',
                   'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets', 'Age',
                   'HospAdmTime', 'ICULOS', 'Gender', 'Unit1', 'Unit2']

x_cols_corr = ['BUN','HR', 'O2Sat', 'Temp', 'SBP', 'DBP', 'Resp', 'BaseExcess', 'FiO2', 'pH',
              'SaO2', 'AST',  'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Magnesium',
              'Phosphate', 'Potassium', 'Bilirubin_total', 'Hgb', 'PTT', 'WBC', 'Platelets', 'Age',
              'HospAdmTime', 'ICULOS', 'Gender', 'Unit1', 'Unit2']

features = ['HR', 'O2Sat', 'Temp', 'SBP', 'Resp', 'pH', 'SaO2', 'Calcium', 'Chloride',
            'Creatinine', 'Magnesium', 'Phosphate', 'Potassium', 'Hgb', 'PTT', 'WBC', 'ICULOS', 'Gender', 'Unit1', 'Unit2']

option1_all_data(features, norm=False)
