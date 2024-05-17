import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Daten laden
file_red_wine = "wine+quality/winequality-red.csv"
file_white_wine = "wine+quality/winequality-white.csv"
data = pd.read_csv(file_red_wine, sep=';')



def plot_data():
    # Explorative Datenanalyse
    # Histogramm für Alkoholgehalt
    plt.hist(data['alcohol'], bins=20, color='skyblue')
    plt.xlabel('Alkoholgehalt in %')
    plt.ylabel('Anzahl') # ToDo: müsste hier nicht auch die Anzahl stehen?
    plt.title('Histogramm des Alkoholgehalts')
    plt.show()

    plt.hist(data['quality'], bins=20, color='skyblue')
    plt.xlabel('Quality')
    plt.ylabel('Anzahl')
    plt.title('Histogramm des Alkoholgehalts')
    plt.show()

    plt.hist(data['residual sugar'], bins=20, color='skyblue')
    plt.xlabel('Zuckergehalt')
    plt.ylabel('g/L')
    plt.title('Histogramm des Zuckergehaltes')
    plt.show()


    plt.boxplot(data['quality'])
    plt.xlabel('Weinqualität 0 - 10')
    plt.title('Boxplot der Weinqualität')
    plt.show()

    plt.scatter(data['chlorides'], data['quality'], alpha=0.5)
    plt.xlabel('Chlorides g/L')
    plt.ylabel('Quality 0 - 10')
    plt.title('Scatterplot: Quality vs. Chlorides')
    plt.show()

    plt.scatter(data['fixed acidity'], data['quality'], alpha=0.5)
    plt.xlabel('Fixed Acidity in g/L')
    plt.ylabel('Quality 0 - 10')
    plt.title('Scatterplot: Quality vs. Acidity')
    plt.show()

plot_data()


scaler = MinMaxScaler()
numerical_features = data.drop('quality', axis=1) # drop non numerical feature quality
normalized_features = scaler.fit_transform(numerical_features)
normalized_data = pd.DataFrame(normalized_features, columns=numerical_features.columns)
normalized_data['quality'] = data['quality'] # add quality again

# Ergebnis anzeigen
print(normalized_data.head())
