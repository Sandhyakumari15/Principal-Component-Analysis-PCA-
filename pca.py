# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:04:03 2020

@author: windows 10
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:00:25 2020

@author: windows 10
"""


import numpy as np 
import pandas as pd 




wine = pd.read_csv("../input/wine-data/wine_data.csv")
wine.head()
wine.tail()
wine.describe()  #describe a data frame ,mean ,median ,mode for integers

#Dropping Index
wine = wine.iloc[:,1:] 
wine.head()

#Normalizing the values
from sklearn.preprocessing import scale 
wine_norm = scale(wine) 
wine_norm

#Building the PCA model

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA()
pca_values = pca.fit_transform(wine_norm)


# The amount of variance that each PCA explains
var = pca.explained_variance_ratio_ 
plt.plot(var)
pd.DataFrame(var)

# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100) 
var1
# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

#storing PCA values to a data frame
new_df = pd.DataFrame(pca_values[:,0:4])
new_df