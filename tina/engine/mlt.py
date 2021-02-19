from tina.engine.path import *
from tina.sampling.random import RandomSampler as MetropolisSampler


@ti.data_oriented
class MLTEngine(metaclass=Singleton):
    def __init__(self):
        self.LSP = 0.3
        self.Sigma = 0.1

        self.nchains = nchains = 2**18
        self.ndims = ndims = 32

        self.X_old = ti.field(float, (nchains, ndims))
        self.X_new = ti.field(float, (nchains, ndims))
        self.L_old = ti.Vector.field(3, float, nchains)
        self.L_new = ti.Vector.field(3, float, nchains)

        ti.materialize_callback(self.reset)

    @ti.kernel
    def reset(self):
        for i in range(self.nchains):
            self.L_old[i] = 0
            for j in range(self.ndims):
                self.X_old[i, j] = ti.random()

    @ti.kernel
    def _inc_film_count(self):
        for i, j in ti.ndrange(FilmTable().nx, FilmTable().ny):
            #impo = 2.0 * (FilmTable().nx * FilmTable().ny) / self.nchains
            impo = 2.0
            FilmTable()[0, i, j] += V(0.0, 0.0, 0.0, impo)

    @ti.func
    def splat(self, x, y, color):
        impo = 0.0
        i, j = ifloor(V(x * FilmTable().nx, y * FilmTable().ny))
        FilmTable()[0, i, j] += V34(color, impo)

    @ti.kernel
    def _render(self):
        for i in range(self.nchains):
            Stack().set(i)

            for j in range(self.ndims):
                self.X_new[i, j] = self.X_old[i, j]

            if ti.random() < self.LSP:
                for j in range(self.ndims):
                    self.X_new[i, j] = ti.random()
            else:
                for j in range(self.ndims):
                    dX = self.Sigma * normaldist(ti.random())
                    self.X_new[i, j] = (self.X_old[i, j] + dX) % 1

            rng = RNGProxy(self.X_new, i)
            ray = Camera().generate(rng.random() * 2 - 1, rng.random() * 2 - 1)
            clr, impo = path_trace(ray, rng)
            self.L_new[i] = clr * impo

            AL_new = Vavg(self.L_new[i]) + 1e-10
            AL_old = Vavg(self.L_old[i]) + 1e-10
            accept = min(1, AL_new / AL_old)
            if accept > 0:
                self.splat(self.X_new[i, 0], self.X_new[i, 1],
                        accept * self.L_new[i] / AL_new)
            self.splat(self.X_old[i, 0], self.X_old[i, 1],
                    (1 - accept) * self.L_old[i] / AL_old)

            if accept > ti.random():
                self.L_old[i] = self.L_new[i]
                for j in range(self.ndims):
                    self.X_old[i, j] = self.X_new[i, j]

            Stack().unset()

    def render(self):
        self._render()
        self._inc_film_count()
