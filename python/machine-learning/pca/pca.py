# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:16:29 2022

@author: emrec
"""

from sklearn.datasets import load_iris
import pandas as pd

#%% 

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target

df = pd.DataFrame(data, columns = feature_names)
df["target"] = y

x = data

#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, whiten=True)  #whiten = normalization
pca.fit(x)

x_pca = pca.transform(x)

print("variance ratio:", pca.explained_variance_ratio_)

print("sum:",sum(pca.explained_variance_ratio_))

#%% 2D visualization

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.target == each],df.p2[df.target == each],color = color[each],label = iris.target_names[each])

plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()











