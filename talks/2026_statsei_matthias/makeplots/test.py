from polyagammapoisson import covariance_kernels as ck
import matplotlib.pyplot as plt
import numpy as np

import scipy.sparse.linalg as sparse_linalg

mx = 101
ny = 57 

x, y = 44, 50
ix, iy = x, ny-y-1

vec = np.zeros((ny, mx))
vec[y,x] = 1

prec = ck.precision_matern(ny, mx, rho=50, v2=1, boundary='')

res = prec @ vec.ravel()

print(res)

plt.figure()

plt.imshow(res.reshape((ny,mx)), origin='lower')

plt.figure()

res = sparse_linalg.spsolve(prec, vec.ravel())   # kernel column of P0^{-1} corresponding to the reference cell

plt.imshow(res.reshape((ny,mx)), origin='lower')


plt.show()