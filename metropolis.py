import numpy as np


def evaluate(X):
    return (X[0] - 0.5)**2 + (X[1] - 0.34)**2


kT = 10

N = 2

X = np.random.rand(N)
E = evaluate(X)

while kT > 0.0005:
    X_old = X
    E_old = E
    dX = np.random.normal(0, 0.1, N)
    X_new = np.clip(X + dX, 0, 1)
    E_new = evaluate(X_new)

    accept = False
    if E_new <= E_old:
        accept = True
    elif np.exp(-(E_new - E_old) / kT) > np.random.rand():
        accept = True
    else:
        accept = False

    if accept:
        X = X_new

    kT *= np.exp(-0.002)

    print(X, E)
