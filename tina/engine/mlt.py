from tina.engine.path import *
from tina.sampling.random import RandomSampler as MetropolisSampler


@ti.data_oriented
class MLTEngine(metaclass=Singleton):
    def __init__(self):
        MetropolisSampler()

    def get_rng(self, i):
        return MetropolisSampler().get_proxy(i)

    def render(self):
        MetropolisSampler().update()
        self._render(2**16)

    @ti.func
    def splat(self, x, y, clr, impo):
        i, j = ifloor(V(x * FilmTable().nx, y * FilmTable().ny))
        FilmTable()[0, i, j] += V34(clr, impo)

    @ti.kernel
    def _render(self, nchain: int):
        for i in range(nchain):
            Stack().set(i)
            rng = self.get_rng(i)

            x, y = random2(rng)
            ray = Camera().generate(x * 2 - 1, y * 2 - 1)

            clr, impo = path_trace(ray, rng)
            self.splat(x, y, clr, impo)

            Stack().unset()
