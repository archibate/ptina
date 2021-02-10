from image import *


@ti.data_oriented
class PathEngine:
    def __init__(self, film):
        self.film = film

    @ti.kernel
    def render(self):
        for i, j in ti.ndrange(self.film.nx, self.film.ny):
            clr = V(i / 512, j / 512, 0.5)
            impo = 1.0
            self.film[i, j] += V34(clr, impo)
