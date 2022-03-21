
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from plot import plot_decision_regions


datos = pd.read_csv('datos.csv')
# aceites = pd.read_csv('aceites.csv', encoding="latin-1")

# Solo las filas con datos completos (para entrenamiento)
completos = datos.dropna()

X = completos.iloc[:, 2:].values
y = completos.iloc[:, 1].values

sc = StandardScaler()
X_train_std = sc.fit_transform(X)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))     
print('Matrix W:\n', w)

X_train_std[0].dot(w)

X_train_pca = X_train_std.dot(w)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']

for key, color in zip(np.unique(y), colors):
    plt.scatter(X_train_pca[y == key, 0], 
                X_train_pca[y == key, 1], 
                c=color, label=key)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)

plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
                    
le = LabelEncoder()
le = le.fit(np.unique(y))

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)

lr = LogisticRegression(max_iter=2000)
lr = lr.fit(X_train_pca, le.transform(y))

print(X_train_pca, le.transform(y))

plot_decision_regions(X_train_pca, le.transform(y), classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
plt.show()