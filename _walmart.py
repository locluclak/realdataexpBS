import pandas as pd 
import numpy as np 
from sklearn.preprocessing import normalize, StandardScaler

def Walmart_sales_holiday(num=20, seed=-1):
    if seed != -1:
        np.random.seed(seed)
    df = pd.read_csv('dataset/Walmart_sales_holiday.csv')

    response = "Weekly_Sales"
    y = df[[response]].to_numpy()
    X = df[[col for col in df.columns if col != response]].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    # X = normalize(X, axis= 0)
    # y = normalize(y, axis= 0)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()
    return X, y

def Walmart_sales_noholiday(num=100, seed=-1):
    if seed != -1:
        np.random.seed(seed)    
    df = pd.read_csv('dataset/Walmart_sales_noholiday.csv')

    response = "Weekly_Sales"
    y = df[[response]].to_numpy()
    X = df[[col for col in df.columns if col != response]].to_numpy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)
    # X = normalize(X, axis= 0)
    # y = normalize(y, axis= 0)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()
    return X, y

if __name__ == "__main__":
    print(Walmart_sales_noholiday()[1])