import pandas as pd 
import numpy as np 
from random import sample
from sklearn.preprocessing import normalize, StandardScaler

def undereq50(num = 20):
    df = pd.read_csv('dataset/Heart_failure_undereq50_data.csv')

    response = "time"
    y = df[[response]].to_numpy()
    X = df[[col for col in df.columns if col != response]].to_numpy()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()

    return X, y

def larger50(num = 100):
    df = pd.read_csv('dataset/Heart_failure_larger50_data.csv')

    response = "time"
    y = df[[response]].to_numpy()
    X = df[[col for col in df.columns if col != response]].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()

    return X, y

if __name__ == "__main__":
    print(larger50()[1])