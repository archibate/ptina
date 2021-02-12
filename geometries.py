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
class HitRecord(namespace):
    @ti.func
    def __init__(self, hit=0, depth=inf, uv=V(0, 0), id=-1):
        self.hit = hit
        self.depth = depth
        self.uv = uv
        self.id = id


@ti.data_oriented
class Face(namespace):
    @ti.func
    def __init__(self, v0, v1, v2, vn0, vn1, vn2, vt0, vt1, vt2):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.vn0 = vn0
        self.vn1 = vn1
        self.vn2 = vn2
        self.vt0 = vt0
        self.vt1 = vt1
        self.vt2 = vt2

    @ti.func
    def true_normal():
        v0, v1, v2 = self.v0, self.v1, self.v2
        return (v1 - v0).cross(v2 - v0).normalized()

    @ti.func
    def normal(self, h):
        u, v = h.uv
        w = V(1 - u - v, u, v)
        vn0, vn1, vn2 = self.vn0, self.vn1, self.vn2
        return (w.x * vn0 + w.y * vn1 + w.z * vn2).normalized()

    @ti.func
    def texcoord(self, h):
        u, v = h.uv
        w = V(1 - u - v, u, v)
        vt0, vt1, vt2 = self.vt0, self.vt1, self.vt2
        return (w.x * vt0 + w.y * vt1 + w.z * vt2)

    @ti.func
    def getbbox(self):
        v0, v1, v2 = self.v0, self.v1, self.v2
        lo = min(v0, v1, v2)
        hi = max(v0, v1, v2)
        return Box(lo, hi)

    @ti.func
    def intersect(self, ray):
        v0, v1, v2 = self.v0, self.v1, self.v2
        ro, rd = ray.o, ray.d
        u = v1 - v0
        v = v2 - v0
        norm = u.cross(v)
        depth = inf * 2
        s, t = 0., 0.
        hit = 0

        b = norm.dot(rd)
        if abs(b) >= eps:
            w0 = ro - v0
            a = -norm.dot(w0)
            r = a / b
            if r > 0:
                ip = ro + r * rd
                uu = u.dot(u)
                uv = u.dot(v)
                vv = v.dot(v)
                w = ip - v0
                wu = w.dot(u)
                wv = w.dot(v)
                D = uv * uv - uu * vv
                s = (uv * wv - vv * wu) / D
                t = (uv * wu - uu * wv) / D
                if 0 <= s <= 1:
                    if 0 <= t and s + t <= 1:
                        depth = r
                        hit = 1
        return HitRecord(hit, depth, V(s, t))
