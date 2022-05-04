
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os

from plot import plot_cluster_values, plot_decision_regions
from regression import LinearRegressionGD

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

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(t_espe_norm)

cov_mat = np.cov(X_train_pca.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

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

    def guardar(self):
        self.estimaciones.to_csv(f'output/{self.nombre}/estimaciones.csv', index=False)
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
    # 'ridge': (),
    # "lasso": Lasso(alpha=0.1),
}



# Todos los espectros normalizados
# Xc = t_espe_norm.values
Xc = X_train_pca

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.scatter(Xc[:, 0], Xc[:, 1])
plt.show()

n_clusters=6
colors = ['blue', 'red', 'green', 'yellow', 'pink', 'orange', 'purple', 'brown', 'black']
#KMeans
km = KMeans(n_clusters=n_clusters, random_state=0)
km.fit(Xc)
kmResult = km.predict(Xc)
labels = km.labels_

esp_clasificados = t_espe_norm.copy()
esp_clasificados = esp_clasificados.assign(TYPE=labels)

def graficar_modelo_clasificado(modelo):
    plt.suptitle("Estimación de las composiciones clasificadas")
    for color, type in zip(colors, range(n_clusters)):
        found = modelo.combinado[modelo.combinado["TYPE"] == type]
        plt.scatter(
            found['G/N'],
            found["P/G"],
            # c=color,
            label=f"Cluster {type}"
        )
    plt.legend()
    plt.xlabel('G/N')
    plt.ylabel('P/G')
    plt.show()

for m in modelos:
    modelo = WrapperModelo(m, modelos[m])
    # Entrenar unicamente utilizando los espectros normalizados (x) y composiciones normalizadas (y) completos
    modelo.fit(X, y.values)
    modelo.ejecutar()
    modelo.estimaciones.insert(loc=1, column="TYPE", value=kmResult)
    modelo.combinado.insert(loc=1, column="TYPE", value=kmResult)
    # modelo.estimaciones.assign(TYPE=kmResult)
    modelo.guardar()
    error = (modelo.combinado.values[c_comp_norm.index][:, 3:6] - modelo.estimaciones.values[c_comp_norm.index][:, 2:5]).astype(float)
    print(error)

    fig, ax = plt.subplots()
    ax.matshow(error)
    for (i, j), z in np.ndenumerate(error):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
    plt.tight_layout()
    plt.show()
    # plt.plot(range(1,modelo.model.n_inter+1), modelo.model.cost_)
    # plt.show()
    # modelo.graficar()
    graficar_modelo_clasificado(modelo)

esp_clasificados = esp_clasificados.assign(ACEITE=datos["WL"])
esp_clasificados.insert(0, "ACEITE", esp_clasificados.pop('ACEITE'))
esp_clasificados.insert(1, "TYPE", esp_clasificados.pop('TYPE'))

cols = [float(x) for x in t_espe_norm.columns]

plt.suptitle('Clasificacion por componentes principales')
plt.title('Método de clustering: KMeans')
plt.xlabel('PC1')
plt.ylabel('PC2')
zipped = np.asarray([x for x in list(zip(kmResult, Xc))])
for label in np.sort(np.unique(labels)):
    coords = np.array([i[1] for i in zipped if i[0] == label])
    plt.scatter(coords[:, 0], coords[:, 1], c=colors[label], label=label)

for i in range(len(km.cluster_centers_)):
    centroid = km.cluster_centers_[i]

plt.legend(loc='upper right')
plot_decision_regions(Xc, labels, classifier=km)

plt.show()

# Xc = t_espe_norm.values

fig, axs = plt.subplots(n_clusters)

esp_clasificados.to_csv("clasificados.csv")

for color, type in zip(colors, range(n_clusters)):
    # rows = esp_clasificados[esp_clasificados["TYPE"] == type].loc[:, esp_clasificados.columns != 'TYPE']
    rows = esp_clasificados[esp_clasificados["TYPE"] == type].iloc[:, 2:]
    axs[type].set_title(f'Cluster {type} ({len(rows)} essential oils)')
    axs[type].set(xlabel='Wavelength', ylabel='Normalized\nAmplitude')
    # Set y lim
    axs[type].set_ylim(0, 1)
    for index, row in rows.iterrows():
        axs[type].plot(
            cols,
            row,
            label=datos['WL'][index],
            # color=color
        )
        
plt.subplots_adjust(
    top=0.96,
    bottom=0.06,
    left=0.125,
    right=0.9,
    hspace=0.89,
    wspace=0.2
)
plt.tight_layout()
plt.savefig('prueba.png', dpi=300)
plt.show()