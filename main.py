
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from plot import plot_decision_regions

def estandarizar(ex):
    return (ex - ex.mean() / ex.std())

def normalizar(ex):
    return (ex - ex.min()) / (ex.max() - ex.min())


datos = pd.read_csv('datos.csv')
# aceites = pd.read_csv('aceites.csv', encoding="latin-1")

completos = datos.dropna()

# Todas las composiciones
t_comp = datos.iloc[:, 2:5].copy()
# Todos los espectros
t_esp = datos.iloc[:, 5:].copy()

# Todos los espectros normalizados
t_espe_norm = normalizar(datos.iloc[:, 5:].copy())

# Composiciones completas normalizadas
c_comp_norm = normalizar(completos.iloc[:, 2:5].copy())
# Espectros completos normalizados
c_espe_norm = t_espe_norm[t_espe_norm.index.isin(c_comp_norm.index)].copy()

# Espectros incompletos normalizados
i_esp_norm = normalizar(datos[datos['C/G'].isna()].iloc[:, 5:].copy())

# plt.matshow(t_espe_norm)
# plt.tight_layout()
# plt.show()

# Solo las filas con datos completos (para entrenamiento)


X = c_espe_norm.values
y = c_comp_norm.values

imr = SimpleImputer(strategy='mean')
# Entrenar con todas las composiciones
imr.fit(t_comp.values)
imputed_data = imr.transform(t_comp.values)
# print(imputed_data)

lr = LogisticRegression(C=100.0, random_state=1)
# lr.fit(X, y)

# plt.scatter(X, y)
# plt.scatter()
