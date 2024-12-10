from sklearn.datasets import load_diabetes
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np
# Define the function to get data with age < 50
def undereq50(num = 20, seed = -1):
    if seed != -1:
        np.random.seed(seed)
    data = load_diabetes(as_frame=True, scaled=False).frame
    filtered_data = data[data['age'] <= 50]
    X = filtered_data.drop(columns=['target']).to_numpy()  # Exclude target column
    y = filtered_data['target'].to_numpy().reshape((-1,1))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()
    return X, y

# Define the function to get data with age >= 50
def larger50(num = 100,seed = -1):
    if seed != -1:
        np.random.seed(seed)
    data = load_diabetes(as_frame=True,scaled=False).frame
    filtered_data = data[data['age'] > 50]
    X = filtered_data.drop(columns=['target']).to_numpy()  # Exclude target column
    y = filtered_data['target'].to_numpy().reshape((-1,1))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    indexes = np.random.choice(y.shape[0], size=num)
    y = y[indexes, :].copy()
    X = X[indexes, :].copy()
    return X, y

if __name__ == "__main__":
    print(undereq50()[0].shape)