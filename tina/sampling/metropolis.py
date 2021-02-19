from tina.sampling import *


@ti.data_oriented
class MetropolisSampler:
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
