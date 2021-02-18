from tina.common import *
from tina.geometries import *


@ti.data_oriented
class LightPool(metaclass=Singleton):
    TYPES = {'POINT': 1, 'AREA': 2}

    def __init__(self, count=2**6):
        self.color = ti.Vector.field(3, float, count)
        self.pos = ti.Vector.field(3, float, count)
        self.size = ti.field(float, count)
        self.type = ti.field(int, count)
        self.count = ti.field(int, ())

    def clear(self):
        self.count[None] = 0

    def add(self, world, color, size, type):
        i = self.count[None]

        pos = world @ np.array([0, 0, 0, 1])
        pos = pos[:3] / pos[3]

        self.type[i] = self.TYPES[type]
        self.color[i] = color.tolist()
        self.pos[i] = pos.tolist()
        self.size[i] = size

        self.count[None] = i + 1
        return i

    @ti.func
    def hit(self, ray):
        ret = namespace(hit=0, dis=inf, pdf=0.0, color=V3(0.0))

        for i in range(self.count[None]):
            type = self.type[i]
            color = self.color[i]
            pos = self.pos[i]
            size = self.size[i]

            t = 0.0
            if type == self.TYPES['POINT']:
                t = Sphere(pos, size**2).intersect(ray)
            elif type == self.TYPES['AREA']:
                t = 2.33

            if 0 < t < ret.dis:
                ret.dis = t
                ret.pdf = ret.dis**2 / (ti.pi * size**2)
                ret.color = color
                ret.hit = 1
                break
        return ret

    @ti.func
    def sample(self, pos, samp):
        i = clamp(ifloor(samp.z * self.count[None]), 0, self.count[None])
        litpos = self.pos[i] + self.size[i] * spherical(samp.x, samp.y)
        toli = litpos - pos

        dis = toli.norm()
        dir = toli / dis
        pdf = dis**2 / (ti.pi * self.size[i]**2)
        color = self.color[i] / pdf
        return namespace(dis=dis, dir=dir, pdf=pdf, color=color)
