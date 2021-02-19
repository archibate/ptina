from tina.common import *


@ti.pyfunc
def trace(X):
    return (X[0] - 0.5)**2 + (X[1] - 0.34)**2


nres = V(64, 64)
film = ti.field(float, nres)
count = ti.field(float, nres)


@ti.func
def splat(X, L):
    I = clamp(ifloor(X * nres), 0, nres - 1)
    film[I] += L
    count[I] += 1


LSP = 0.1

nchains = 1024
ndims = 2


Xs = ti.Vector.field(ndims, float, nchains)
Ls = ti.field(float, nchains)


@ti.kernel
def render():
    for i in range(nchains):
        X_old = Xs[i]
        L_old = Ls[i]

        X_new = X_old
        if i == 0 or ti.random() < LSP:
            X_new = random2(ti)
        else:
            dX = 0.1 * normaldist(random2(ti))
            X_new = (X_old + dX) % 1

        L_new = trace(X_new)

        AL_new = Vavg(L_new) + 1e-10
        AL_old = Vavg(L_old) + 1e-10
        accept = min(1, AL_new / AL_old)
        if accept > 0:
            splat(X_new, accept * L_new / AL_new)
        splat(X_old, 1 - accept * L_old / AL_old)

        if accept > ti.random():
            L_old = L_new
            X_old = X_new

        Xs[i] = X_old
        Ls[i] = L_old


gui = ti.GUI()

while gui.running and not gui.get_event(gui.ESCAPE):
    render()
    gui.set_image(ti.imresize(film.to_numpy() / (count.to_numpy() + 1e-10), 512))
    gui.show()
