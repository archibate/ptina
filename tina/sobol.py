from tina.common import *
try:
    from pysobol import Sobol as NumpySobol
except ImportError:
    please_install('pysobol')


@ti.func
def wanghash(x):
    value = ti.cast(x, ti.u32)
    value = (value ^ 61) ^ (value >> 16)
    value *= 9
    value ^= value << 4
    value *= 0x27d4eb2d
    value ^= value >> 15
    return int(value)


@ti.func
def wanghash2(x, y):
    value = wanghash(x)
    value = wanghash(y ^ value)
    return value


@ti.func
def wanghash3(x, y, z):
    value = wanghash(x)
    value = wanghash(y ^ value)
    value = wanghash(z ^ value)
    return value


@ti.func
def unixfasthash(x):
    value = ti.cast(x, ti.u32)
    value = (value * 7**5) % (2**31 - 1)
    return int(value)


@ti.pyfunc
def binaryreverse(i):
    j = 0
    k = 1
    while i != 0:
        k <<= 1
        j <<= 1
        j |= i & 1
        i >>= 1
    return j / k


@ti.data_oriented
class TaichiSobol:
    def __init__(self, dim=21201, nsamples=2**20, skip=17):
        self.dim = dim
        self.nsamples = nsamples
        self.skip = skip
        self.data = ti.field(float, self.dim)
        self.reset()

    def reset(self):
        self.core = NumpySobol(self.nsamples, self.dim)
        for i in range(self.skip):
            next(self.core)

    def update(self):
        arr = next(self.core)
        self.data.from_numpy(arr)

    @ti.func
    def calc(self, x):
        return self.data[x % self.dim]

    def get_proxy(self, x):
        return self.Proxy(self, x)

    @ti.data_oriented
    class Proxy:
        def __init__(self, sobol, x):
            self.sobol = sobol
            self.x = ti.expr_init(x)
            self.y = ti.expr_init(0)

        @ti.func
        def random(self):
            ret = self.sobol.calc(wanghash2(self.x, self.y))
            self.y += 1
            return ret


if __name__ == '__main__':
    n = 128
    sobol1 = TaichiSobol(1024)
    sobol2 = TaichiSobol(1024)
    img1 = ti.field(float, (n, n))
    img2 = ti.field(float, (n, n))
    img3 = ti.field(float, (n, n))


    @ti.kernel
    def render_image():
        for i, j in ti.ndrange(n, n):
            id = wanghash2(i, j)
            so = sobol1.get_proxy(id)
            img1[i, j] += so.random()
            img2[i, j] += sobol2.calc(id)
            img3[i, j] += ti.random()


    gui1 = ti.GUI('sobol+hash')
    gui2 = ti.GUI('sobol')
    gui3 = ti.GUI('pseudo')
    gui1.fps_limit = 10
    gui2.fps_limit = 10
    gui3.fps_limit = 10
    while gui1.running and gui2.running and gui3.running:
        sobol1.update()
        sobol2.update()
        render_image()
        gui1.set_image(ti.imresize(img1.to_numpy() / (1 + gui1.frame), 512))
        gui2.set_image(ti.imresize(img2.to_numpy() / (1 + gui2.frame), 512))
        gui3.set_image(ti.imresize(img3.to_numpy() / (1 + gui3.frame), 512))
        gui1.show()
        gui2.show()
        gui3.show()


__all__ = ['wanghash', 'wanghash2', 'wanghash3', 'TaichiSobol']
