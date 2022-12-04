# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 10:21:50 2022

@author: emrec
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial_regression.csv", sep=";")

x = df.araba_fiyat.values.reshape(-1,1)
y = df.araba_max_hiz.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.show()

# linear regression = y = b0 + b1*x
# multiple linear regression = y = b0 + b1*x1 + b2*x2

#%% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% predict

y_head = lr.predict(x)
plt.scatter(x,y)
plt.xlabel("araba_fiyat")
plt.ylabel("araba_max_hiz")
plt.plot(x,y_head,color="red", label="linear")
plt.legend()
plt.show()

test = lr.predict([[10000]])
 
"""
but max_speed cant be 871, so this model doesnt fit to our data
we need to use polynomial linear regression here
"""

# %% 

# polynomial linear regression = y = b0 + b1*x + b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4)

x_polynomial = polynomial_regression.fit_transform(x)

#%% fit

linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

#%% 

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2, color= "black", label = "poly")
plt.legend()
plt.show()






















