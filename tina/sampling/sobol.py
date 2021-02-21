'''
Sobol quasi-random generator for low occupancy sequence

references: https://web.maths.unsw.edu.au/~fkuo/sobol/
'''

from tina.sampling import *
try:
    # https://github.com/archibate/pysobol
    from pysobol import Sobol as NumpySobol
except ImportError:
    please_install('pysobol')


@ti.data_oriented
class SobolSampler(metaclass=Singleton):
    def __init__(self, dim=21201, nsamples=2**20, skip=64):
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

        @ti.func
        def random(self):
            ret = self.sobol.calc(self.x)
            self.x += 1
            return ret


if __name__ == '__main__':
    n = 128
    sobol = SobolSampler(1024)
    img1 = ti.Vector.field(3, float, (n, n))
    img2 = ti.Vector.field(3, float, (n, n))


    @ti.kernel
    def render_image():
        for i, j in ti.ndrange(n, n):
            so = sobol.get_proxy(wanghash2(i, j))
            img1[i, j] += V(so.random(), so.random(), so.random())
            img2[i, j] += V(ti.random(), ti.random(), ti.random())


    gui1 = ti.GUI('sobol')
    gui2 = ti.GUI('pseudo')
    gui2.fps_limit = gui1.fps_limit = 5
    while gui1.running and gui2.running:
        sobol.update()
        render_image()
        gui1.set_image(ti.imresize(img1.to_numpy() / (1 + gui1.frame), 512))
        gui2.set_image(ti.imresize(img2.to_numpy() / (1 + gui2.frame), 512))
        gui1.show()
        gui2.show()
