import pandas as pd
import random

def add_std(val, col):
    l1 = [1, 1.5, -1, -1.5, 2, -2]
    x = random.choice(l1)
    return val + x*data_std[col]

orig_path = "full_df_mean_train.psv"
train_df = pd.read_csv("full_df_mean_train.psv", delimiter='|')
train_df1 = train_df[train_df['label'] == 1]
data_std = train_df1.std()

for c in train_df1.columns:
    train_df1[c] = train_df1[c].apply(add_std, col=c)


full_df = pd.concat([train_df, train_df1])
full_df.to_csv("full_df_mean_train_duplicate_std_2.psv", sep='|')
