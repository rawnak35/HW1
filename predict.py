import pickle
import sys
import pandas as pd
import os
import binaryClassificationMean
from sklearn.preprocessing import StandardScaler
import torch




directory = sys.argv[1]
path_out = "test.csv"
model_path = 'model.pt' #TODO

i = 0
for filename in os.listdir(directory):
    full_path = directory + "/" + filename
    df = pd.read_csv(full_path, delimiter="|")
    # print(df.head())
    SepsisLabel = df['SepsisLabel'].to_list()
    try:
        idx = SepsisLabel.index(1)
        label = 1
    except:
        idx = len(SepsisLabel)
        label = 0
    df = df.truncate(after=idx)
    df['label'] = label

    df = df.mean(axis=0).to_frame().T
    df['patient'] = int(filename[8:-4])

    if i == 0:
        full_df = df.__deepcopy__()
        i += 1
    else:
        full_df = pd.concat([full_df, df])
full_df.fillna(0, inplace=True)
full_df.to_csv(path_out, sep="|")
# ----------------------------------------------------------

features = ['HR', 'O2Sat', 'Temp', 'SBP', 'Resp', 'pH', 'SaO2', 'Calcium', 'Chloride','Creatinine',
            'Magnesium', 'Phosphate', 'Potassium', 'Hgb', 'PTT', 'WBC', 'ICULOS', 'Gender', 'Unit1', 'Unit2']

X_test = full_df[features]
# load the normalization model (trained on the training data)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test = scaler.transform(X_test)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_data = binaryClassificationMean.TestData(torch.FloatTensor(X_test))
test_loader = binaryClassificationMean.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

model = binaryClassificationMean.BinaryClassification()

model.load_state_dict(torch.load(model_path))
model.eval()
y_pred_list = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())


y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

data_out = pd.DataFrame({"Id": full_df['patient'].tolist(), "SepsisLabel": y_pred_list})
data_out.to_csv('prediction.csv', header=False, index=False)

