from common import *


@ti.data_oriented
class Ray(namespace):
    @ti.func
    def __init__(self, o, d):
        self.o = o
        self.d = d


@ti.data_oriented
class Box(namespace):
    @ti.func
    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    @ti.func
    def intersect(self, r):
        near = 0.0
        far = inf
        hit = 1

        for i in ti.static(range(3)):
            if abs(r.d[i]) < eps:
                if r.o[i] < self.lo[i] or r.o[i] > self.hi[i]:
                    hit = 0
            else:
                i1 = (self.lo[i] - r.o[i]) / r.d[i]
                i2 = (self.hi[i] - r.o[i]) / r.d[i]

                far = min(far, max(i1, i2))
                near = max(near, min(i1, i2))

                if near > far:
                    hit = 0

        return namespace(hit=hit, near=near, far=far)


@ti.data_oriented
class Face(namespace):
    @ti.func
    def __init__(self, v0, v1, v2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2

