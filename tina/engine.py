from tina.image import *
from tina.camera import *
from tina.model import *
from tina.light import *
from tina.materials import *
from tina.acceltree import *
from tina.stack import *


@ti.func
def power_heuristic(a, b):
    a = clamp(a, eps, inf)**2
    b = clamp(b, eps, inf)**2
    return a / (a + b)


@ti.data_oriented
class PathEngine(metaclass=Singleton):
    def __init__(self):
        self.bgm = Image.load('assets/env.png')
        self.film = Image.new(512, 512)

    @ti.func
    def trace(self, r):
        avoid = -1
        depth = 0
        result = V3(0.0)
        importance = 1.0
        throughput = V3(1.0)
        last_brdf_pdf = 0.0

        #material = Phong()
        #material = Disney()
        material = Lambert()

        while depth < 4 and Vany(throughput > eps) and importance > eps:
            depth += 1

            r.d = r.d.normalized()
            hit = BVHTree().intersect(r, avoid)

            #'''
            lit = LightPool().hit(r)
            if hit.hit == 0 or lit.dis < hit.depth:
                mis = power_heuristic(last_brdf_pdf, lit.pdf)
                direct_li = mis * lit.color
                result += throughput * direct_li
            #'''

            if hit.hit == 0:
                #result += throughput * self.bgm(*dir2tex(r.d)).xyz
                break

            avoid = hit.index
            face = ModelPool().get_face(hit.index)
            normal = face.normal(hit)
            hitpos = r.o + hit.depth * r.d

            #'''
            li = LightPool().sample(hitpos, random3())
            occ = BVHTree().intersect(Ray(hitpos, li.dir), avoid)
            if occ.hit == 0 or occ.depth > li.dis:
                brdf_clr = material.brdf(normal, -r.d, li.dir)
                brdf_pdf = Vavg(brdf_clr)
                mis = power_heuristic(li.pdf, brdf_pdf)
                direct_li = mis * li.color * brdf_clr * dot_or_zero(normal, li.dir)
                result += throughput * direct_li
            #'''

            brdf = material.bounce(normal, -r.d, random3())
            importance *= brdf.impo
            throughput *= brdf.color
            r.o = hitpos
            r.d = brdf.outdir
            last_brdf_pdf = brdf.pdf

        return result, importance

    @ti.kernel
    def render(self):
        camera = Camera(V(0.0, 0.0, 4.8))
        for i, j in ti.ndrange(self.film.nx, self.film.ny):
            Stack().set(i * self.film.nx + j)

            dx, dy = 0.5, 0.5
            x = (i + dx) / self.film.nx * 2 - 1
            y = (j + dy) / self.film.ny * 2 - 1
            ray = camera.generate(x, y)
            clr, impo = self.trace(ray)
            self.film[i, j] += V34(clr, impo)

            Stack().unset()

    def get_image(self, hdr=False):
        return self.film.to_numpy_normalized(ToneMapping() if not hdr else None)
