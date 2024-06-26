from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    # markers = ('s', 'x', 'o', '^', 'v')
    colors = ('blue', 'red', 'green', 'yellow', 'pink', 'orange', 'purple', 'brown', 'black')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black',
                    # marker=markers[idx], 
                    label=cl)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


colors = ['blue', 'red', 'green', 'yellow', 'pink', 'orange', 'purple', 'brown', 'black']

def plot_cluster_values(df, n_clusters, columns, labels, colors=colors):
    fig, axs = plt.subplots(n_clusters)
    for color, type in zip(colors, range(n_clusters)):
        rows = df[df["TYPE"] == type].iloc[:, 2:]
        axs[type].set_title(f'KMeans Cluster {type} ({len(rows)} aceites)')
        axs[type].set(xlabel='Longitud de onda', ylabel='Valor')
        for index, row in rows.iterrows():
            axs[type].plot(
                columns,
                row,
                label=labels[index],
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