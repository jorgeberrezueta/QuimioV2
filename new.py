import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from plot import plot_decision_regions

from regression import LinearRegressionGD

#################
### DEFINICIONES
#################

ratios = ['C/G', 'G/N', 'P/G']

models = []

np.set_printoptions(precision=10, suppress=True)

def estandarizar(ex):
    return (ex - ex.mean() / ex.std())

def normalizar(ex):
    return (ex - ex.min()) / (ex.max() - ex.min())

#################
### CARGAR DATOS
#################

datos = pd.read_csv('datos.csv')

# print(datos)

datos_norm = datos.copy()
datos_norm.loc[:, datos_norm.select_dtypes('number').columns] = normalizar(datos_norm.loc[:, datos_norm.select_dtypes('number').columns])

# print(datos_norm)

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

############################################
### ANALISIS ESTADISTICO PARA CLASIFICACION
############################################

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(t_espe_norm)

# print("Componentes principales: ")
# print(X_train_pca)

cov_mat = np.cov(X_train_pca.T)

# print("Matriz de covarianza: ")
# print(cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# print("Eigen valores: ")
# print(eig_vals)

# print("Eigen vectores: ")
# print(eig_vecs)

Xc = X_train_pca

plt.title("Principal Component Analysis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.scatter(Xc[:, 0], Xc[:, 1])
plt.xlim(min(Xc[:, 0]) - min(Xc[:, 0]) * 0.05, max(Xc[:, 0]) + max(Xc[:, 0]) * 0.05)
plt.ylim(min(Xc[:, 1]) - min(Xc[:, 1]) * 0.05, max(Xc[:, 1]) + max(Xc[:, 1]) * 0.05)
plt.show()

############################
# CLASIFICACION CON K-MEANS
############################

# CLASIFICACION POR COMPONENTES PRINCIPALES
n_clusters=6
colors = ['blue', 'red', 'green', 'yellow', 'pink', 'orange', 'purple', 'brown', 'black']
km = KMeans(n_clusters=n_clusters, random_state=0)
km.fit(X_train_pca)
kmResult = km.predict(X_train_pca)
labels = km.labels_

print(kmResult)

for label, color in zip(np.unique(labels), colors):
    plt.scatter(
        Xc[kmResult == label, 0],
        Xc[kmResult == label, 1],
        color=color,
        label=label
    )
    
plt.title("Principal Component Analysis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plot_decision_regions(Xc, labels, classifier=km)
plt.xlim(min(Xc[:, 0]) - min(Xc[:, 0]) * 0.05, max(Xc[:, 0]) + max(Xc[:, 0]) * 0.05)
plt.ylim(min(Xc[:, 1]) - min(Xc[:, 1]) * 0.05, max(Xc[:, 1]) + max(Xc[:, 1]) * 0.05)
plt.show()

##############
### EJECUCION
##############

# Por cada columna, entrenar un modelo de regresion
for r in ratios:
    model = LinearRegressionGD(n_iter=20)
    model.fit(
        c_espe_norm,
        c_comp_norm[r], 
    )
    models.append(model)

# Por cada modelo entrenado, graficar el error
for r, wmodel in zip(ratios, models):
    # Graficar el error del modelo
    plt.plot(
        range(1, model.n_iter + 1),  
        model.cost_, 
        label=r
    )
plt.ylabel('SSE')
plt.xlabel('Epoch')
if (model.n_iter < 20):
    plt.xticks(range(1, model.n_iter + 1), range(1, model.n_iter + 1))
# plt.legend()
plt.show()

###################
### GENERAR TABLAS
###################

if os.path.exists(f'output/linear') is False:
    os.makedirs(f'output/linear')
# Obtener una estimacion utilizando todos los espectros normalizados
data = np.array([m.predict(t_espe_norm) for m in models])
data = data.transpose()
print(data)
estimaciones = pd.DataFrame(columns=["est_C/G", "est_G/N", "est_P/G"], data=data)
estimaciones.insert(loc=0, column="WL", value=datos["WL"])

# REEMPLAZAR VALORES EN LA MATRIZ ORIGINAL
combinado = datos_norm.copy()
indices_vacios = combinado[combinado["C/G"].isna()].index
combinado['STATE'] = 'original'
combinado.loc[combinado["C/G"].isna(), 'STATE'] = 'estimacion'
combinado.loc[combinado["C/G"].isna(), 'C/G'] = estimaciones.values[:, 1][indices_vacios]
combinado.loc[combinado["G/N"].isna(), 'G/N'] = estimaciones.values[:, 2][indices_vacios]
combinado.loc[combinado["P/G"].isna(), 'P/G'] = estimaciones.values[:, 3][indices_vacios]
combinado = combinado[["WL", "STATE", "C/G", "G/N", "P/G"]]
print(combinado)

estimaciones.insert(loc=1, column="TYPE", value=kmResult)
combinado.insert(loc=1, column="TYPE", value=kmResult)

estimaciones.to_csv(f'output/linear/estimaciones.csv', index=False)
combinado.to_csv(f'output/linear/combinado.csv', index=False)


##########################################
### GRAFICAR LOS CLUSTERS INDIVIDUALMENTE
##########################################

cols = [float(x) for x in t_espe_norm.columns]

fig, axs = plt.subplots(n_clusters)

for color, type in zip(colors, range(n_clusters)):
    # Clasificacion por componentes principales
    rows = t_espe_norm[kmResult == type]
    axs[type].set_title(f'Cluster {type} ({len(rows)} essential oils)')
    axs[type].set(xlabel='Wavelength', ylabel='Normalized\nAmplitude')
    # Set y lim
    axs[type].set_ylim(0, 1)
    for index, row in rows.iterrows():
        axs[type].plot(
            cols,
            row,
            label=datos['WL'][index]
        )
        
plt.subplots_adjust(
    top=0.96,
    bottom=0.06,
    left=0.125,
    right=0.9,
    hspace=0.89,
    wspace=0.2
)
# plt.tight_layout()
plt.savefig('prueba.png', dpi=300)
plt.show()