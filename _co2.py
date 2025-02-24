import pandas as pd 
import numpy as np 
from sklearn.preprocessing import normalize, StandardScaler

def gasoline_fuel(num=20,seed=-1):
    if seed != -1:
        np.random.seed(seed)
    df = pd.read_csv('dataset/CO2_gasoline_fuel.csv')

    response = "CO2 Emissions(g/km)"
    y = df[[response]].to_numpy()
    X = df[[col for col in df.columns if col != response]].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # y = scaler.fit_transform(y)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()
    return X, y

def other_fuel(num=100,seed=-1):
    if seed != -1:
        np.random.seed(seed)
    df = pd.read_csv('dataset/CO2_other_fuel.csv')

    response = "CO2 Emissions(g/km)"
    y = df[[response]].to_numpy()
    X = df[[col for col in df.columns if col != response]].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # y = scaler.fit_transform(y)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()
    return X, y

if __name__ == "__main__":
    print(other_fuel()[0])