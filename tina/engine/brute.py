from tina.engine import *


@ti.func
def power_heuristic(a, b):
    a = clamp(a, eps, inf)**2
    b = clamp(b, eps, inf)**2
    return a / (a + b)


@ti.data_oriented
class BruteEngine(metaclass=Singleton):
    def __init__(self):
        DefaultSampler()

    def get_rng(self, i, j):
        return DefaultSampler().get_proxy(wanghash2(i, j))

    def render(self):
        DefaultSampler().update()
        self._render()

    @ti.func
    def trace(self, r, rng):
        avoid = -1
        depth = 0
        result = V3(0.0)
        importance = 1.0
        throughput = V3(1.0)

        while depth < 5 and Vany(throughput > eps) and importance > eps:
            depth += 1

            r.d = r.d.normalized()
            hit = BVHTree().intersect(r, avoid)

            lit = LightPool().hit(r)
            if lit.hit != 0 and (hit.hit == 0 or lit.dis < hit.depth):
                result += throughput * lit.color

            if hit.hit == 0:
                result += throughput * 0.05
                break

            avoid = hit.index
            hitpos, normal, sign, material = ModelPool().get_geometries(hit, r)

            sign = -r.d.dot(normal)
            if sign < 0:
                normal = -normal

            brdf = material.bounce(normal, sign, -r.d, random3(rng))
            importance *= brdf.impo
            throughput *= brdf.color
            r.o = hitpos
            r.d = brdf.outdir

        return result, importance

    @ti.kernel
    def _render(self):
        for i, j in ti.ndrange(FilmTable().nx, FilmTable().ny):
            Stack().set(i * FilmTable().nx + j)
            rng = self.get_rng(i, j)

            dx, dy = random2(rng)
            x = (i + dx) / FilmTable().nx * 2 - 1
            y = (j + dy) / FilmTable().ny * 2 - 1
            ray = Camera().generate(x, y)

            clr, impo = self.trace(ray, rng)
            FilmTable()[0, i, j] += V34(clr, impo)

            Stack().unset()
