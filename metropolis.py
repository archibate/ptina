from tina.common import *


@ti.pyfunc
def trace(X):
    return (X[0] - 0.5)**2 + (X[1] - 0.34)**2


M = 64
film = ti.field(float, (M, M))
count = ti.field(float, (M, M))


@ti.func
def splat(X, L):
    I = ifloor(X * M)
    film[I] += L
    count[I] += 1


LSP = 0.1

@ti.kernel
def render():
    for _ in range(16):
        X = V2(0.0)
        L = 0.0

        for i in range(M * M):
            X_old = X
            L_old = L

            X_new = X
            if i == 0 or ti.random() < LSP:
                X_new = random2(ti)
            else:
                dX = 0.1 * normaldist(random2(ti))
                X_new = (X + dX) % 1

            L_new = trace(X_new)

            AL_new = Vavg(L_new) + 1e-10
            AL_old = Vavg(L_old) + 1e-10
            accept = min(1, AL_new / AL_old)
            if accept > 0:
                splat(X_new, accept * L_new / AL_new)
            splat(X_old, 1 - accept * L_old / AL_old)

            if accept > ti.random():
                L = L_new
                X = X_new
            else:
                L = L_old
                X = X_old


render()
ti.imshow(ti.imresize(film.to_numpy() / (count.to_numpy() + 1e-10), 512))
