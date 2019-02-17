import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# dunno what this does
# %dmatplotlib inline

# must load all datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

np.random.seed(1)

X, Y = noisy_moons
X, Y = X.T, Y.reshape(1, Y.shape[0])
print(Y)

suM = [1,2,3,4,0]
try:
    sum(suM)
except TypeError:
    print("Not a number")