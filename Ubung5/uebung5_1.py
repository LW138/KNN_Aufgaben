from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def encode_data(data):
    # one-hot encoding for categorical variables
    data_encoded = pd.get_dummies(data, drop_first=True)
    return data_encoded

def fill_missing_values(data):
    # fill missing values with mean for missing values
    data_filled = data.fillna(data.mean())
    return data_filled

def normalisieren(data):
    # create a scaler
    scaler = MinMaxScaler()

    # use the scaler to transform the data
    # fit_transform is a combination of fit and transform
    # fit calculates the mean, variance, min and max (MinMaxScaler) etc.
    # transform uses these values to transform the data
    data_scaled = scaler.fit_transform(data)

    # reform the data into a pandas DataFrame
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

if __name__ == '__main__':
    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets

    # metadata
    print("Adult Metadata: ", adult.metadata)

    # variable information
    print("Adult Variables: ", adult.variables)

    # use functions for data preprocessing
    # makes a one hot encoding on categorical data
    X_encoded = encode_data(X)
    X_filled = fill_missing_values(X_encoded)
    X_scaled = normalisieren(X_filled)

    print("X Head: ", X.head())

    print("X Encdoded: ", X_encoded.head())

    print("X filled head: ", X_filled.head())

    print("X scaled: ", X_scaled.head())

