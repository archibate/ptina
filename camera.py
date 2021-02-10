from geometries import *


@ti.data_oriented
class Camera(namespace):
    is_taichi_class = True

    @ti.func
    def __init__(self, pos=V(0.0, 0.0, 4.0),
            up=V(0.0, 1.0, 0.0), tgt=V3(0.0), fov=45.0):
        self.pos = pos
        self.fwd = (tgt - pos).normalized()
        self.right = self.fwd.cross(up).normalized()
        self.up = self.right.cross(self.fwd)
        self.fov = 1 / ti.tan(fov * ti.pi / 360)

    @ti.func
    def generate(self, x, y):
        ro = self.pos
        rd = (self.right * x + self.up * y + self.fwd * self.fov).normalized()
        return Ray(ro, rd)
