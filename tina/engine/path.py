from tina.engine import *
from tina.sampling import *
from tina.sampling.sobol import *


@ti.func
def power_heuristic(a, b):
    a = clamp(a, eps, inf)**2
    b = clamp(b, eps, inf)**2
    return a / (a + b)


@ti.func
def path_trace(r, rng):
    avoid = -1
    depth = 0
    result = V3(0.0)
    throughput = V3(1.0)
    last_brdf_pdf = 0.0

    while depth < 5 and Vany(throughput > 0) and Vany(r.d != 0):
        depth += 1

        r.d = r.d.normalized()
        hit = BVHTree().intersect(r, avoid)

        lit = LightPool().hit(r)
        if lit.hit != 0 and (hit.hit == 0 or lit.dis < hit.depth):
            mis = power_heuristic(last_brdf_pdf, lit.pdf)
            direct_li = mis * lit.color
            result += throughput * direct_li

        if hit.hit == 0:
            result += throughput * WorldLight().at(r.d)
            break

        avoid = hit.index
        hitpos, normal, sign, material = ModelPool().get_geometries(hit, r)

        sign = -r.d.dot(normal)
        if sign < 0:
            normal = -normal

        li = LightPool().sample(hitpos, random3(rng))
        occ = BVHTree().intersect(Ray(hitpos, li.dir), avoid)
        if occ.hit == 0 or occ.depth > li.dis:
            brdf_clr = material.brdf(normal, sign, -r.d, li.dir)
            brdf_pdf = Vavg(brdf_clr)
            mis = power_heuristic(li.pdf, brdf_pdf)
            direct_li = mis * li.color * brdf_clr * dot_or_zero(normal, li.dir)
            result += throughput * direct_li

        brdf = material.bounce(normal, sign, -r.d, random3(rng))
        throughput *= brdf.color
        r.o = hitpos
        r.d = brdf.outdir
        last_brdf_pdf = brdf.pdf

    return result


@ti.data_oriented
class PathEngine(metaclass=Singleton):
    def __init__(self):
        SobolSampler()

    def get_rng(self, i, j):
        return SobolSampler().get_proxy(wanghash2(i, j))

    def render(self):
        SobolSampler().update()
        self._render()

    @ti.kernel
    def _render(self):
        for i, j in ti.ndrange(FilmTable().nx, FilmTable().ny):
            Stack().set(i * FilmTable().nx + j)
            rng = self.get_rng(i, j)

            dx, dy = random2(rng)
            x = (i + dx) / FilmTable().nx * 2 - 1
            y = (j + dy) / FilmTable().ny * 2 - 1
            ray = Camera().generate(x, y)

            clr = path_trace(ray, rng)
            FilmTable()[0, i, j] += V34(clr, 1.0)

            Stack().unset()
