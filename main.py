
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

lr = LinearRegression()

# Entrenar unicamente utilizando los espectros normalizados (x) y composiciones normalizadas (y) completos
lr.fit(X, y)

# Obtener una estimacion utilizando todos los espectros normalizados
estimaciones = pd.DataFrame(columns=["est_C/G", "est_G/N", "est_P/G"], data=lr.predict(t_espe_norm))
estimaciones.insert(loc=0, column="WL", value=datos["WL"])
estimaciones.to_csv('output/estimaciones.csv', index=False)

#############################################
### REEMPLAZAR VALORES EN LA MATRIZ ORIGINAL
#############################################

combinado = datos.copy()

indices_vacios = combinado[combinado["C/G"].isna()].index

combinado['STATE'] = 'original'
combinado.loc[combinado["C/G"].isna(), 'STATE'] = 'estimacion'
combinado.loc[combinado["C/G"].isna(), 'C/G'] = estimaciones.values[:, 0][indices_vacios]
combinado.loc[combinado["G/N"].isna(), 'G/N'] = estimaciones.values[:, 1][indices_vacios]
combinado.loc[combinado["P/G"].isna(), 'P/G'] = estimaciones.values[:, 2][indices_vacios]

combinado = combinado[["WL", "STATE", "C/G", "G/N", "P/G"]]

print(combinado)

combinado.to_csv('output/combinado.csv', index=False)

#############
### GRAFICAR
#############

# Graficar X = G/N, Y = C/G, color = STATE

plt.scatter(
    combinado[combinado["STATE"] == 'original']['G/N'],
    combinado[combinado["STATE"] == "original"]["P/G"],
    c='blue',
)
plt.scatter(
    combinado[combinado["STATE"] == 'estimacion']['G/N'],
    combinado[combinado["STATE"] == "estimacion"]["P/G"],
    c='red',
)

plt.xlabel('G/N')
plt.ylabel('P/G')

plt.show()

# plt.matshow(t_espe_norm)
# plt.show()
