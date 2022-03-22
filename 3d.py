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

colX = t_comp.iloc[:,1] # G/N
colY = t_comp.iloc[:,2]  # P/G
colZ = t_comp.iloc[:,0]  # C/G 

for t in np.unique(completos.iloc[:,1].values):
    ax.scatter(
        colX, # G/N
        colY,  # P/G
        colZ # C/G 
    )

plt.show()