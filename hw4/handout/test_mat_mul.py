import numpy as np

X = np.arange(0,9).reshape(3,3)
v = np.arange(0,3)
print(X)
print(v)
u = np.zeros(X.shape[0])
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        u[i] += X[i, j] * v[j]

# print("u is : {}".format(u))
# res1 = X @ v
# print("res1 is : {}".format(res1))

# res2 = np.matmul(X, v)
# print("res2 is : {}".format(res2))

# res3 = u = np.dot(X, v)
# print("res3 is : {}".format(res3))


# res4 = u = np.dot(v, X)
# print("res4 is : {}".format(res4))

# res5 = np.matmul(v, X)
# print("res5 is : {}".format(res5))

# res6 = X * v
# print("res6 is : {}".format(res6))


# res7 = v * X
# print("res7 is : {}".format(res7))


# res8 = np.dot(v, X)
# print("res8 is : {}".format(res8))




# # [[-3.  0.  3.]
# #  [-3.  0.  3.]
# #  [-3.  0.  3.]]
# print((X - np.mean(X, axis=0)).T)
# # [[-1.  0.  1.]
# #  [-1.  0.  1.]
# #  [-1.  0.  1.]]
# print(X - np.mean(X, axis=1, keepdims=True))
# # [[-3. -3. -3.]
# #  [ 0.  0.  0.]
# #  [ 3.  3.  3.]]
# print(X - np.mean(X, axis=0, keepdims=True))
# # [[-1.  0.  1.]
# #  [-1.  0.  1.]
# #  [-1.  0.  1.]]
# print(X - np.expand_dims(np.mean(X, axis=1), 1))

w = np.arange(3,6)
#print(w)
M = X - np.mean(X, axis=1, keepdims=True).T

print(np.matmul(w * M.T, M))
print(np.matmul(w * M, M.T))
print(np.dot(w * M, M.T))
print(w * np.dot(M.T, M))
print("t")
print(np.dot(np.dot(w, M.T), M))
