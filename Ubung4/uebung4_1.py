import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyse_data(title, data):
    # Einen ersten Blick auf die Daten werfen
    print(data.head())
    print(data.describe())

    # Fehlende Werte überprüfen und behandeln
    print(data.isnull().sum())
    data = data.fillna(data.mean())
    print(data.isnull().sum())

    # Korrelationsmatrix berechnen
    korrelationen = data.corr(method='pearson')
    print(korrelationen)

    # Heatmap der Korrelationsmatrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(korrelationen, annot=True, cmap='coolwarm', center=0)
    plt.title('Heatmap der Korrelationen: ' + title)
    plt.show()

# CSV-Datei einlesen
data_red = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
data_white = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

analyse_data('Red Wine', data_red)
analyse_data('White Wine', data_white)

