import numpy as numpy



X = np.array(np.arange(9))

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        u[i] += X[i, j] * v[j]