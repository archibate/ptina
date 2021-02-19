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


X_old = ti.Vector.field(ndims, float, nchains)
X_new = ti.Vector.field(ndims, float, nchains)
L_old = ti.field(float, nchains)
L_new = ti.field(float, nchains)


@ti.kernel
def render():
    for i in range(nchains):
        X_new[i] = X_old[i]
        if i == 0 or ti.random() < LSP:
            X_new[i] = random2(ti)
        else:
            dX = 0.1 * normaldist(random2(ti))
            X_new[i] = (X_old[i] + dX) % 1

        L_new[i] = trace(X_new[i])

        AL_new = Vavg(L_new[i]) + 1e-10
        AL_old = Vavg(L_old[i]) + 1e-10
        accept = min(1, AL_new / AL_old)
        if accept > 0:
            splat(X_new[i], accept * L_new[i] / AL_new)
        splat(X_old[i], 1 - accept * L_old[i] / AL_old)

        if accept > ti.random():
            L_old[i] = L_new[i]
            X_old[i] = X_new[i]


gui = ti.GUI()

while gui.running and not gui.get_event(gui.ESCAPE):
    render()
    gui.set_image(ti.imresize(film.to_numpy() / (count.to_numpy() + 1e-10), 512))
    gui.show()
