import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df_ratings = pd.read_csv('ratings.csv', header=None)
df_ratings.drop(1, axis=1, inplace=True)
df_ratings.set_index(0, inplace=True)
df_ratings.index.name = 'ID'

ratings = df_ratings.replace(0, np.NaN).mean()
ratings = (ratings - 1) / 4
S = len(ratings)  # Numero totale di canzoni

# Fit del polinomio di terzo grado
X = np.arange(1, len(ratings) + 1).reshape(-1, 1)
y = np.array(ratings)

poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Calcolo della probabilità p per ogni canzone in base al ranking
p_values = []
for rank in range(1, len(ratings) + 1):
    rank_poly = poly_features.transform([[rank]])
    p = poly_model.predict(rank_poly)[0]
    p_values.append(p)

print(p_values)

# Sostituisci i valori di p al posto della relazione lineare nel tuo codice

#%% Load listens data

# df = pd.read_csv('downloads_exp1_independent.csv', header=None)
df = pd.read_csv('listens_exp1_independent.csv', header=None)
df.drop(1, axis=1, inplace=True)
df.set_index(0, inplace=True)
df.index.name = 'ID'
df = df.loc[~(df == 0).all(axis=1)]

d = df.T.sum(axis=1)  # Somma per i partecipanti
D = d.sum()  # Somma anche per le canzoni

m = d / D  # Frazione di download o ascolti
S = len(m)  # Numero totale di canzoni

listening_probability = d / 200

#%%

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()  # Tutti i valori sono trattati allo stesso modo, gli array devono essere 1D
    if np.amin(array) < 0:
        array -= np.amin(array)  # I valori non possono essere negativi
    array += 0.0000001  # I valori non possono essere 0
    array = np.sort(array)  # I valori devono essere ordinati
    index = np.arange(1, array.shape[0] + 1)  # Indice per ogni elemento dell'array
    n = array.shape[0]  # Numero di elementi nell'array
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Coefficiente di Gini

#%% Calcola Gini

# gini_indices = []
# for i in range(8):
#     # Carica i dati degli ascolti dal file del mondo corrente
#     df_world = pd.read_csv(f'Social influence/dynamics_listens_w1_v1.txt{i + 1}.csv', header=None)
#     df_world.drop(1, axis=1, inplace=True)
#     df_world.set_index(0, inplace=True)
#     df_world.index.name = 'ID'
#     df_world = df_world.loc[~(df_world == 0).all(axis=1)]

#     d_world = df_world.T.sum(axis=1)  # Somma per i partecipanti
#     D_world = d_world.sum()  # Somma anche per le canzoni

#     m_world = d_world / D_world  # Frazione di ascolti
#     gini_index_world = gini(np.array(m_world))
#     gini_indices.append(gini_index_world)

# mean_gini = np.mean(gini_indices)
# std_gini = np.std(gini_indices)
