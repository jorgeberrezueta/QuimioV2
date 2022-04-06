from random import randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = []
y = []

x += [200 + randint(-100,100) for i in range(100)]
y += [800 + randint(-100,100) for i in range(100)]

x += [500 + randint(-100,100) for i in range(100)]
y += [200 + randint(-100,100) for i in range(100)]

x += [200 + randint(-100,100) for i in range(100)]
y += [500 + randint(-100,100) for i in range(100)]

c = [[x[i], y[i]] for i in range(len(x))]

# plt.scatter(x, y)
# plt.show()

colors = ["red", "blue", "green", "purple", "orange", "yellow", "black", "brown", "pink", "gray", "cyan"]

km = KMeans(n_clusters=5, random_state=0)
km.fit(c)
kmResult = km.predict(c)
labels = km.labels_

centroids = km.cluster_centers_

for i in range(len(c)):
    plt.scatter(
        c[i][0], 
        c[i][1], 
        color=colors[kmResult[i]]
    )

for centroid in centroids:
    plt.scatter(
        centroid[0], 
        centroid[1], 
        color="black", 
        marker="x", 
        s=25
    )
plt.show()