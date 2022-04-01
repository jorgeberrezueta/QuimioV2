
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
import os

from plot import plot_decision_regions

np.set_printoptions(precision=10, suppress=True)

def estandarizar(ex):
    return (ex - ex.mean() / ex.std())

def normalizar(ex):
    return (ex - ex.min()) / (ex.max() - ex.min())

#################
### CARGAR DATOS
#################

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

#################################################
# CREAR MODELO DE REGRESION, ENTRENAR Y PREDECIR
#################################################

# Espectros completos normalizados
X = c_espe_norm.values
# Composiciones completas normalizadas
y = c_comp_norm

# plt.matshow(t_espe_norm)
# plt.show()

class WrapperModelo():
    # Crear constructor
    def __init__(self, nombre, model):
        self.nombre = nombre
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def ejecutar(self):
        if os.path.exists(f'output/{self.nombre}') is False:
            os.makedirs(f'output/{self.nombre}')
        # Obtener una estimacion utilizando todos los espectros normalizados
        self.estimaciones = pd.DataFrame(columns=["est_C/G", "est_G/N", "est_P/G"], data=self.predict(t_espe_norm))
        self.estimaciones.insert(loc=0, column="WL", value=datos["WL"])
        self.estimaciones.to_csv(f'output/{self.nombre}/estimaciones.csv', index=False)
        #############################################
        ### REEMPLAZAR VALORES EN LA MATRIZ ORIGINAL
        #############################################
        self.combinado = datos.copy()
        self.indices_vacios = self.combinado[self.combinado["C/G"].isna()].index
        self.combinado['STATE'] = 'original'
        self.combinado.loc[self.combinado["C/G"].isna(), 'STATE'] = 'estimacion'
        self.combinado.loc[self.combinado["C/G"].isna(), 'C/G'] = self.estimaciones.values[:, 1][self.indices_vacios]
        self.combinado.loc[self.combinado["G/N"].isna(), 'G/N'] = self.estimaciones.values[:, 2][self.indices_vacios]
        self.combinado.loc[self.combinado["P/G"].isna(), 'P/G'] = self.estimaciones.values[:, 3][self.indices_vacios]
        self.combinado = self.combinado[["WL", "STATE", "C/G", "G/N", "P/G"]]
        print(self.combinado)
        self.combinado.to_csv(f'output/{self.nombre}/combinado.csv', index=False)

    def graficar(self):
        #############
        ### GRAFICAR
        #############
        # Graficar X = G/N, Y = C/G, color = STATE
        plt.scatter(
            self.combinado[self.combinado["STATE"] == 'original']['G/N'],
            self.combinado[self.combinado["STATE"] == "original"]["P/G"],
            c='blue',
        )
        plt.scatter(
            self.combinado[self.combinado["STATE"] == 'estimacion']['G/N'],
            self.combinado[self.combinado["STATE"] == "estimacion"]["P/G"],
            c='red',
        )
        plt.xlabel('G/N')
        plt.ylabel('P/G')
        plt.show()

modelos = {
    "linear": LinearRegression(),
    "lasso": Lasso(alpha=0.1),
}

for m in modelos:
    modelo = WrapperModelo(m, modelos[m])
    # Entrenar unicamente utilizando los espectros normalizados (x) y composiciones normalizadas (y) completos
    modelo.fit(X, y)
    modelo.ejecutar()
    # modelo.graficar()  