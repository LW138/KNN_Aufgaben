from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

def encode_data(data):
    # One-Hot-Encoding für kategorische Spalten
    data_encoded = pd.get_dummies(data, drop_first=True)
    return data_encoded

def fill_missing_values(data):
    # Fehlende Werte mit dem Durch
    data_filled = data.fillna(data.mean())
    return data_filled

def normalisieren(data):
    # Erstellen Sie den Skalierer
    scaler = MinMaxScaler()  # oder StandardScaler()

    # Passen Sie den Skalierer an die Daten an und transformieren Sie sie
    data_scaled = scaler.fit_transform(data)

    # Umwandeln in DataFrame
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    return data_scaled

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features
y = adult.data.targets

# metadata
print(adult.metadata)

# variable information
print(adult.variables)

# Funktionen für die Datenvorbereitung
X_encoded = encode_data(X)
X_filled = fill_missing_values(X_encoded)
X_scaled = normalisieren(X_filled)

print(X.head())

print(X_encoded.head())

print(X_filled.head())

print(X_scaled.head())

