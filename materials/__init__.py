from common import *


@ti.data_oriented
class BSDFSample(namespace):
    def __init__(self, outdir=V3(0.0), pdf=0.0, color=V3(0.0)):
        self.outdir = outdir
        self.pdf = pdf
        self.color = color


@ti.data_oriented
class Mirror(namespace):
    def __init__(self, color=V3(1.0)):
        self.color = color

    @ti.func
    def brdf(self, normal, indir, outdir):
        return V3(0.0)

    @ti.func
    def bounce(self, normal, indir, samp):
        outdir = reflect(-indir, normal)
        return BSDFSample(outdir, inf, self.color)


@ti.data_oriented
class Lambert(namespace):
    def __init__(self, color=V3(1.0)):
        self.color = color

    @ti.func
    def brdf(self, normal, indir, outdir):
        cosi = dot_or_zero(indir, normal)
        coso = dot_or_zero(outdir, normal)
        return self.color / ti.pi

    @ti.func
    def bounce(self, normal, indir, samp):
        outdir = tanspace(normal) @ spherical(ti.sqrt(samp.x), samp.y)
        return BSDFSample(outdir, 1 / ti.pi, self.color)
