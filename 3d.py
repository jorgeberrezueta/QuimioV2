import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datos = pd.read_csv('datos.csv')
# aceites = pd.read_csv('aceites.csv', encoding="latin-1")

completos = datos.dropna()

# Todas las composiciones
t_comp = datos.iloc[:, 2:5].copy()

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

colors = ["red", "blue", "green", "purple", "orange", "yellow"]

for c, t in zip(colors, np.unique(completos.iloc[:,1].values)):
    ax.scatter(
        completos[completos["TYPE"] == t].iloc[:,2], # G/N
        completos[completos["TYPE"] == t].iloc[:,3], # P/G
        completos[completos["TYPE"] == t].iloc[:,4], # C/G
        c=c,
        edgecolors='black'
    )

ax.legend(np.unique(completos.iloc[:,1].values))

plt.show()