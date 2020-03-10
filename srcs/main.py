import sys
import numpy as np
import pandas as pd
sys.path.append('../utils')
from FileLoader import FileLoader
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from function_tools import *
from accuracy_score import accuracy_score_

def normalize(df):
    df_norm = pd.DataFrame()
    try:
        for col in df:
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    except:
        print("Error: value in data feature is not a float")
        exit()
    return df_norm

if __name__ == "__main__":
    if len(sys.argv) == 3:
        loader = FileLoader()
        data_train = loader.load(str(sys.argv[1]))
        data_test = loader.load(str(sys.argv[2]))
        # prep data
        x_train = data_train.drop(columns=['Index','diagnosis'])
        x_norm = normalize(x_train)
        x_train = x_norm.to_numpy()
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = data_test.drop(columns=['Index','diagnosis'])
        x_norm = normalize(x_test)
        x_test = x_norm.to_numpy()
        y_true = data_test.drop(columns=['Index','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']).to_numpy()
        y_true = np.where(y_true == 'M', 1, 0)
        y = data_train.drop(columns=['Index','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']).to_numpy()
        y_train = np.where(y == 'M', 1, 0)
        # network
        net = Network()
        net.add(FCLayer(30, 15))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(15, 10))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(10, 1))
        net.add(ActivationLayer(tanh, tanh_prime))
        #train
        net.use(mse, mse_prime)
        net.fit(x_train, y_train, epochs=100, learning_rate=0.1)
        #test
        out = net.predict(x_test)
        print(out)
        y_pred = [1 if x > 0.5 else 0 for x in out]
        print("accuracy : ", accuracy_score_(y_pred, y_true))
    else:
        print("Usage : python Network.py train.csv test.csv")