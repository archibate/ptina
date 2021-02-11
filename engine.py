from texture import *
from camera import *


@ti.data_oriented
class PathEngine:
    def __init__(self, film, geom):
        self.bgm = Texture.load('assets/env.png')
        self.geom = geom
        self.film = film

    @multireturn
    @ti.func
    def trace(self, r):
        yield V3(0.0)

        hit = self.tree.intersect(r)
        if hit.hit == 0:
            yield gammize(self.bgm(*dir2tex(r.d)).xyz)

        yield V3(1.0)

    @ti.kernel
    def render(self):
        camera = Camera()
        for i, j in ti.ndrange(self.film.nx, self.film.ny):
            dx, dy = 0.5, 0.5
            x = (i + dx) / self.film.nx * 2 - 1
            y = (j + dy) / self.film.ny * 2 - 1
            ray = camera.generate(x, y)
            clr = self.trace(ray)
            self.film[i, j] += V34(clr, 1.0)
