import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler



#Read Data

orig_path = "full_df_mean_train.psv"
train_df = pd.read_csv("full_df_mean_train_duplicate_std.psv", delimiter='|')
train_df =train_df.fillna(0)
test_df = pd.read_csv("full_df_mean_test.psv", delimiter='|')

test_df = test_df.fillna(0)

features = ['HR', 'O2Sat', 'Temp', 'SBP', 'Resp', 'pH', 'SaO2', 'Calcium', 'Chloride',
            'Creatinine', 'Magnesium', 'Phosphate', 'Potassium', 'Hgb', 'PTT', 'WBC', 'ICULOS', 'Gender', 'Unit1', 'Unit2']


target = ['label']
X_train = train_df[features]
train_df = train_df.apply(pd.to_numeric)
y_train = train_df[target]

X_test = test_df[features]
test_df = test_df.apply(pd.to_numeric)
y_test = test_df[target]

#Standardize Input
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
with open("scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
X_test = scaler.transform(X_test)


#Model Parameters
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
feature_size = 20


## train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        print(type(X_data))
        print("====================")
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

## test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 31.
        self.layer_1 = nn.Linear(feature_size, BATCH_SIZE)
        self.layer_2 = nn.Linear(BATCH_SIZE, BATCH_SIZE)
        self.layer_out = nn.Linear(BATCH_SIZE, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(BATCH_SIZE)
        self.batchnorm2 = nn.BatchNorm1d(BATCH_SIZE)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


def train():
    """
    train the NN model
    :return: None
    """
    train_data = TrainData(torch.FloatTensor(X_train),torch.FloatTensor(y_train.values))
    test_data = TrainData(torch.FloatTensor(X_test), torch.FloatTensor(y_test.values))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryClassification()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_list_train = []
    loss_list_test = []
    f1_list_train = []
    f1_list_test = []
    model.train()

    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_loss_test = 0
        prob_train = []
        prob_test = []
        y_pred_all = np.array([]).astype(int)
        y_true_all = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            y_pred_tag = torch.round(torch.sigmoid(y_pred))
            y = y_batch.detach().numpy().astype(int)
            y = np.concatenate(y).ravel()
            y_pred_tag = y_pred_tag.detach().numpy().astype(int)
            y_pred_tag = np.concatenate(y_pred_tag).ravel()
            y_true_all = np.concatenate((y_true_all, y), axis=None)
            y_pred_all = np.concatenate((y_pred_all, y_pred_tag), axis=None)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        f1 = f1_score(y_pred_all, y_true_all)
        f1_list_train.append(f1)
        loss_list_train.append(epoch_loss / len(train_loader))
        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | F1 score: {f1:.3f}')
        y_pred_list = []
        y_model_list = []
        model.eval()
        if e % 1 == 0:
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_test_pred = model(X_batch)
                    loss_ = criterion(y_test_pred, y_batch)
                    epoch_loss_test += loss_.item()
                    y_model_list.append(y_test_pred.cpu().numpy())
                    y_test_pred = torch.sigmoid(y_test_pred)
                    y_pred_tag = torch.round(y_test_pred)
                    y_pred_list.append(y_pred_tag.cpu().numpy())


            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

            f1_ = f1_score(y_pred_list, y_test)

            loss_list_test.append(epoch_loss_test/len(test_loader))
            f1_list_test.append(f1_)
            print("###############Test#############")
            print("F1 score  ", f1_)
            print("Loss: ", epoch_loss_test/len(test_loader))
            print('------------------------------')
    data_out = pd.DataFrame({"Id": test_df['patient'].tolist(), "SepsisLabel": y_pred_list})
    data_out.to_csv('prediction.csv', header=False, index=False)
    torch.save(model.state_dict(), 'model_3.pt')

    with open('loss_list_train.pkl', 'wb') as f:
        pickle.dump(loss_list_train, f)

    with open('loss_list_test.pkl', 'wb') as f:
        pickle.dump(loss_list_test, f)

    with open('f1_list_train.pkl', 'wb') as f:
        pickle.dump(f1_list_train, f)

    with open('f1_list_test.pkl', 'wb') as f:
        pickle.dump(f1_list_test, f)


def predict(model_path):
    """
    load the model and predict, and save results
    :param model_path: the NN model
    :return: None
    """
    model = BinaryClassification()

    model.load_state_dict(torch.load(model_path))

    train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train.values))
    test_data = TrainData(torch.FloatTensor(X_test), torch.FloatTensor(y_test.values))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    data = [train_loader, test_loader]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y_test_true = []
    y_prob_test = []
    y_train_true = []
    y_prob_train = []
    model.eval()
    for i in range(2):
        with torch.no_grad():
            for X_batch, y_batch in data[i]:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                if i == 0:
                    y_prob_train.append(torch.sigmoid(y_pred))
                    y_train_true.append(y_batch)
                else:
                    y_prob_test.append(torch.sigmoid(y_pred))
                    y_test_true.append(y_batch)

    with open('y_test_true.pkl', 'wb') as f:
        pickle.dump(y_test_true, f)

    with open('y_prob_test.pkl', 'wb') as f:
        pickle.dump(y_prob_test, f)
    with open('y_train_true.pkl', 'wb') as f:
        pickle.dump(y_train_true, f)

    with open('y_prob_train.pkl', 'wb') as f:
        pickle.dump(y_prob_train, f)


# to train the model run train()
# train()
