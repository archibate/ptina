import numpy as np
import timeit


import numpy as np


# https://web.maths.unsw.edu.au/~fkuo/sobol/
def Sobol(N, D):
    f = open('/home/bate/Downloads/new-joe-kuo-6.21201')
    f.readline()

    L = int(np.ceil(np.log2(N)))

    C = np.full(N, 0)
    C[0] = 1
    for i in range(N):
        C[i] = 1
        value = i
        while value & 1:
            value >>= 1
            C[i] += 1

    V = np.full((L + 1, D), 0)

    for j in range(D):
        if j != 0:
            _, s, a, M = f.readline().split(maxsplit=3)
            s, a = int(s), int(a)
            m = np.full(s + 1, 0)
            for i, M in enumerate(map(int, M.split())):
                m[i + 1] = M
        else:
            m = np.full(L + 1, 1)
            s = L

        if L <= s:
            for i in range(L + 1):
                V[i, j] = m[i] << (32 - i)
        else:
            for i in range(s + 1):
                V[i, j] = m[i] << (32 - i)
            for i in range(s + 1, L + 1):
                V[i, j] = V[i - s, j] ^ (V[i - s, j] >> s)
                for k in range(1, s):
                    V[i, j] ^= ((a >> (s - 1 - k)) & 1) * V[i - k, j]

    X = np.full(D, 0)
    for i in range(N):
        P = X / 2**32
        X = X ^ V[C[i], :]
        yield P


from tina.common import *


@ti.data_oriented
class SobolRNG(metaclass=Singleton):
    def __init__(self, xdim, ydim, zdim, nsamples=8192, skip=1, jitter=False):
        dim = xdim * ydim * zdim
        self.nsamples = nsamples
        self.core = Sobol(nsamples, dim)
        for i in range(skip):
            next(self.core)
        self.data = ti.field(float, dim)
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.jitter = jitter

    def update(self):
        arr = next(self.core)
        self.data.from_numpy(arr)

    @ti.func
    def get(self, x, y, z):
        x = x % self.xdim
        y = y % self.ydim
        z = z % self.zdim
        return self.data[(x * self.ydim + y) * self.zdim + z]


n = 128
rng = SobolRNG(n, n, 1)
img1 = ti.field(float, (n, n))
img2 = ti.field(float, (n, n))


@ti.kernel
def render_image():
    for i, j in ti.ndrange(n, n):
        img1[i, j] += rng.get(i, j, 0)
        img2[i, j] += ti.random()


gui1 = ti.GUI('sobol')
gui2 = ti.GUI('pseudo')
gui1.fps_limit = 10
gui2.fps_limit = 10
while gui1.running and gui2.running:
    rng.update()
    render_image()
    gui1.set_image(ti.imresize(img1.to_numpy() / (1 + gui1.frame), 512))
    gui2.set_image(ti.imresize(img2.to_numpy() / (1 + gui2.frame), 512))
    gui1.show()
    gui2.show()
