import numpy as np
import taichi as ti


def erfinv(x):
    sgn = np.sign(x)

    x = (1 - x) * (1 + x)
    lnx = np.log(x)

    tt1 = 2 / (np.pi * 0.147) + 0.5 * lnx
    tt2 = 1 / 0.147 * lnx

    return sgn * np.sqrt(-tt1 + np.sqrt(tt1**2 - tt2))


def normal(samp):
    return np.sqrt(2) * erfinv(samp * 2 - 1)


'''
import matplotlib.pyplot as plt

plt.hist(normal(np.random.rand(4096))
plt.show()

exit(0)
'''


def trace(X):
    return (X[0] - 0.5)**2 + (X[1] - 0.34)**2


M = 64
film = np.zeros((M, M))
count = np.zeros((M, M)) + 1e-10

def splat(X, L):
    I = tuple(np.int32(np.floor(X * M)))
    film[I] += L
    count[I] += 1


N = 2

X = np.random.rand(N)
L = trace(X)

LSP = 0.1


for i in range(M * M * 8):
    large = np.random.rand() < LSP

    X_old = X
    L_old = L

    if large:
        X_new = np.random.rand(N)
    else:
        #dX = 0.1 * normal(np.random.rand(N))
        dX = np.random.normal(0, 0.1, N)
        X_new = (X + dX) % 1

    L_new = trace(X_new)

    accept = min(1, np.average(L_new / L_old))
    if accept > 0:
        splat(X_new, accept * L_new / np.average(L_new))
    splat(X_old, 1 - accept * L_old / np.average(L_old))

    if accept > np.random.rand():
        L = L_new
        X = X_new
    else:
        L = L_old
        X = X_old


ti.imshow(ti.imresize(film / count, 512))
