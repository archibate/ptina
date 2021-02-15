import numpy as np
import timeit


data = np.load('/tmp/sobol.npy')
POLY = data[:, 0]
V_TMPL = data[:, 1:]
DIM, LOG = V_TMPL.shape[:2]
print(f'[TinaRNG] Sobol data size: DIM={DIM}, LOG={LOG}')


N2 = 0xfffffffe
#N2 = 0xffffffff & -(1 << (31 - LOG))


def GetHighestBitPos(n):
    # Returns the position of the high 1 bit base 2 in an integer.
    bit = 0
    while n > 0:
        bit += 1
        n >>= 1
    return bit


def GetLowestBitPos(n):
    # Returns the position of the low 0 bit base 2 in an integer.
    bit = 1
    while n & 1:
        bit += 1
        n >>= 1
    return bit


def GenVG(dim):
    vg = np.full((dim, LOG), 0)

    for i in range(dim):
        for j in range(LOG):
            vg[i, j] = V_TMPL[i, j]

    for i in range(2, dim + 1):
        m = GetHighestBitPos(POLY[i - 1] >> 1)

        inc = np.full(m, 0)

        l = POLY[i - 1]
        for k in range(m, 0, -1):
            inc[k - 1] = l & 1
            l >>= 1

        for j in range(m + 1, LOG + 1):
            newv = vg[i - 1, j - m - 1]
            l = 1
            for k in range(1, m + 1):
                l <<= 1
                if inc[k - 1]:
                    newv ^= l * vg[i - 1, j - k - 1]
            vg[i - 1, j - 1] = newv

    l = 1
    for j in range(LOG - 1, 0, -1):
        l <<= 1
        for k in range(dim):
            vg[k, j - 1] *= l

    print(vg)
    return vg


class Sobol:
    def __init__(self, dim, nsamples, skip):
        assert dim <= DIM, dim
        assert nsamples <= 2**LOG, nsamples

        self.dim = dim
        self.nsamples = nsamples
        self.vg = GenVG(self.dim)
        self.seed = 0
        self.last_q = np.full(self.dim, 0)
        self.last_sample = np.full(self.dim, -1)
        self.nshift = GetHighestBitPos(nsamples)
        assert self.nshift != 0

        for i in range(skip):
            self.nextImpl()

        assert skip == self.seed

    def nextImpl(self):
        l = GetLowestBitPos(self.seed)
        if LOG <= l:
            self.seed = 0
            l = 0
        else:
            self.seed += 1

        self.last_sample = self.last_q >> (LOG + 1 - self.nshift)
        self.last_q ^= self.vg[:, l]

        return np.any(self.nsamples < self.last_sample)

    def next(self):
        while self.nextImpl():
            print('invalid sample encountered!')
        return self.last_sample

    def nextFloat(self):
        return (self.next() + 0.5) / self.nsamples


#'''
rng = Sobol(32, 1024, 173)
print(rng.next())
exit(1)
#'''


from tina.common import *


@ti.data_oriented
class SobolRNG(metaclass=Singleton):
    def __init__(self, xdim, ydim, zdim, nsamples=8192, skip=173, jitter=False):
        dim = xdim * ydim * zdim
        self.nsamples = nsamples
        self.core = Sobol(dim, nsamples, skip)
        self.data = ti.field(int, dim)
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.jitter = jitter

    def generate(self):
        arr = self.core.next()
        self.data.from_numpy(arr)

    @ti.func
    def get(self, x, y, z):
        x = x % self.xdim
        y = y % self.ydim
        z = z % self.zdim
        val = self.data[(x * self.ydim + y) * self.zdim + z]
        if ti.static(self.jitter):
            return (val + ti.random()) / self.nsamples
        else:
            return (val + 0.5) / self.nsamples

    @ti.func
    def get3(self, x, y):
        u = self.get(x, y, 0)
        v = self.get(x, y, 1)
        w = self.get(x, y, 2)
        return V(u, v, w)


rng = SobolRNG(32, 32, 1)
img1 = ti.field(float, (32, 32))
img2 = ti.field(float, (32, 32))


@ti.kernel
def render_image():
    for i, j in ti.ndrange(32, 32):
        img1[i, j] += rng.get(i, j, 0)
        img2[i, j] += ti.random()


gui1 = ti.GUI('sobol')
gui2 = ti.GUI('pseudo')
gui1.fps_limit = 10
gui2.fps_limit = 10
while gui1.running and gui2.running:
    rng.generate()
    render_image()
    gui1.set_image(ti.imresize(img1.to_numpy() / (1 + gui1.frame), 512))
    gui2.set_image(ti.imresize(img2.to_numpy() / (1 + gui2.frame), 512))
    gui1.show()
    gui2.show()
