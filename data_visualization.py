import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def hist_with_label(column_name, data_path="full_df_mean_train.psv"):
    """
    build and save histogram of column in tow the groups: label=1, label=0
    :param column_name: column name in data frame
    :return: None
    """
    #data_path = "full_df_mean_train.psv"
    data = pd.read_csv(data_path, delimiter='|')
    print(data.columns)
    ax = data[data['label'] == 0][column_name].plot.hist(bins=30, alpha=0.5, color='red')
    ax = data[data['label'] == 1][column_name].plot.hist(bins=30, alpha=0.5)
    ax.legend(["NOT Sepsis", "Sepsis"])
    ax.set_title(column_name)
    ax.set_ylabel("Frequency")
    ax.set_xlabel('Value')
    ax.figure.savefig("plots/plots_with_labels_length")
    ax.figure.clf()


def bar_plot_frequency(data, col):
    """
    build and save histogram of *binary* column in tow the groups: label=1, label=0
    :param data: dataframe
    :param col: column name
    :return: None
    """
    ax = data.groupby(col)['label'].value_counts().unstack().plot.bar(rot=0,alpha=0.5, color= ['red', 'blue'])
    ax.set_title(col)
    ax.legend(["NOT Sepsis", "Sepsis"])
    ax.set_ylabel("Frequency")
    ax.figure.savefig("plots/plots_with_labels_mean_new"+"/bar_" + col)
    ax.figure.clf()


def scatter_plot(data, col1, col2):
    """
    build and save scatter plot: plot of col1 vs. col2
    :param data: dataframe pandas
    :param col1: column 1 name
    :param col2: column 2 name
    :return: None
    """
    l1 = data[col1].tolist()
    l2 = data[col2].tolist()
    plt.scatter(l1, l2)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f"plot of {col1} vs. {col2}")
    plt.savefig("plots/corr_new" + "/" + col1 + '_' + col2)
    plt.clf()


def box_plot(data, col):
    """
    build and save box plot for col values, with label=1, and label=1
    :param data: dataframe - pandas
    :param col: column name
    :return:None
    """
    data = data[[col, 'label']].dropna()
    l1 = data[data['label'] == 0][col].tolist()
    l2 = data[data['label'] == 1][col].tolist()
    dict_ = {'Not Sepsis': l1, 'Sepsis': l2}
    fig, ax = plt.subplots()
    ax.boxplot(dict_.values())
    ax.set_xticklabels(dict_.keys())
    plt.title(col)
    plt.savefig("plots/box_plots" + "/drop_outliers_" + col)
    plt.clf()


def drop_outliers(data, col, threshold, op):
    """
    remove outliers from column
    :param data: dataframe
    :param col: column name
    :param threshold: threshold drop_outliers
    :param op: operator : '<' or '>'
    :return: new dataframe after removing outliers
    """
    df = data.copy()
    if op == '>':
        df = df[df[col] > threshold]
    else:
        df = df[df[col] < threshold]
    return df


def plot_NN_eval(list_train, list_test, name):
    """
    plot the loss / f1 NN model
    :param list_train: training values - list
    :param list_test: test values - list
    :param name: F1-score OR Loss
    :return: None
    """
    plt.plot(list_train, label='Train')
    plt.plot(list_test, label='Test')
    plt.legend()
    plt.title("Model " + name)
    plt.savefig("plots/NN/" + name)
    plt.clf()
def ROC(y_true, y_probas, name):
    print(len(y_true))
    print(len(y_probas))

    #skplt.metrics.plot_roc_curve(y_true, y_probas)


    fpr, tpr, _ = roc_curve(y_true, y_probas)
    roc_auc = roc_auc_score(y_true, y_probas)
    plt.figure(1)
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr, label="CNN(area={: .3f})".format(roc_auc))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve - " + name)
    plt.savefig("plots/" + "ROC curve_" + name)
    plt.clf()

if __name__ == '__main__':

    # plots
    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
               'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose', 'Magnesium',
               'Phosphate', 'Potassium', 'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets', 'Age',
               'HospAdmTime', 'ICULOS']
    for c in columns:
        hist_with_label(c)


    with open('loss_list_train.pkl', 'rb') as f:
        loss_list_train = pickle.load(f)

    with open('loss_list_test.pkl', 'rb') as f:
        loss_list_test = pickle.load(f)

    with open('f1_list_train.pkl', 'rb') as f:
        f1_list_train = pickle.load(f)

    with open('f1_list_test.pkl', 'rb') as f:
        f1_list_test = pickle.load(f)

    #plot_NN_eval(loss_list_train, loss_list_test, "Loss")
    plot_NN_eval(f1_list_train, f1_list_test, ' F1-score')
    print('train', f1_list_train[-1])
    print('test', f1_list_test[-1])

  
    with open('y_test_true.pkl', 'rb') as f:
        y_test_true = pickle.load(f)

    with open('y_prob_test.pkl', 'rb') as f:
        y_prob_test=  pickle.load( f)
    with open('y_train_true.pkl', 'rb') as f:
        y_train_true = pickle.load(f)

    with open('y_prob_train.pkl', 'rb') as f:
        y_prob_train = pickle.load(f)

    y_test_true = [x.tolist()[0] for x in y_test_true]
    y_prob_test = [x.tolist()[0] for x in y_prob_test]
    y_train_true = [x.tolist()[0] for x in y_train_true]
    y_prob_train = [x.tolist()[0] for x in y_prob_train]


    print("y_true_list\n", y_test_true[:5])
    print("y_prpb_test\n", y_prob_test[:5])
    ROC(y_test_true, y_prob_test, "Test")
    ROC(y_train_true, y_prob_train, "Train")


    data_path = "full_df_mean_train.psv"
    data = pd.read_csv(data_path, delimiter='|')


# scatter plot
    with open('cols_list_scatter.pkl', 'rb') as f:
        cols = pickle.load(f)

    for c in cols:
        scatter_plot(data, c[0], c[1])

    # box plots
    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'HospAdmTime', 'ICULOS']
    for c in columns:
        box_plot(data, c)

    # hist for binary features
    columns = ['Gender', 'Unit1', 'Unit2']
    data_path = "full_df_mean_train.psv"
    data = pd.read_csv(data_path, delimiter='|')
    for c in columns:
        bar_plot_frequency(data, c)

"""    # remove outliers and build box, scatter plots
    columns = ['HospAdmTime', 'Resp', 'O2Sat']
    thresholds = [(-4000, '>'), (50, '<'), (60, '>')]

    for c, th in zip(columns, thresholds):
        data_path = "full_df_mean_train.psv"
        data = pd.read_csv(data_path, delimiter='|')
        df = drop_outliers(data, c, th[0], th[1])
        bar_plot(df, c)
        if c == "O2Sat":
            scatter_plot(df, "O2Sat", "Temp")
           
            print(f"corr test after removing outliers { res}")
"""
