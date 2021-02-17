from tina.image import *


@ti.data_oriented
class FilmTable(metaclass=Singleton):
    is_taichi_class = True

    def __init__(self, size=2**22, count=1):  # 64 MB
        self.res = ti.Vector.field(2, int, ())
        self.root = ti.Vector.field(4, float, (count, size))

    @property
    @ti.pyfunc
    def nx(self):
        return self.res[None].x

    @property
    @ti.pyfunc
    def ny(self):
        return self.res[None].y

    @nx.setter
    @ti.pyfunc
    def nx(self, value):
        self.res[None].x = nx

    @ny.setter
    @ti.pyfunc
    def ny(self, value):
        self.res[None].y = ny

    @ti.func
    def subscript(self, id, x, y):
        index = x * self.ny + y
        return self.root[id, index]

    def set_size(self, nx, ny):
        self.res[None] = nx, ny

    def clear(self, id=0):
        self.root.fill(0)

    def to_numpy(self, id=0, tonemap=NoToneMap):
        arr = np.empty((self.nx, self.ny, 3), np.float32)
        self._to_numpy(id, arr, tonemap)
        return arr

    @ti.kernel
    def _to_numpy(self, id: int, arr: ti.ext_arr(), tonemap: ti.template()):
        nx, ny = self.res[None]
        for x, y in ti.ndrange(nx, ny):
            val = self[id, x, y]
            if val.w != 0:
                val.xyz /= val.w
            else:
                val.xyz = V(0.9, 0.4, 0.9)
            val = tonemap(val)
            for k in ti.static(range(3)):
                arr[x, y, k] = val[k]

    def get_image(self, id=0, raw=False):
        return self.to_numpy(id, ToneMapping() if not raw else None)

    @ti.kernel
    def _fast_export_image(self, id: int, out: ti.ext_arr()):
        shape = self.res[None]
        for x, y in ti.ndrange(shape.x, shape.y):
            base = (y * shape.x + x) * 3
            I = V(x, y)
            val = self[id, x, y]
            if val.w != 0:
                val.xyz /= val.w
            else:
                val.xyz = V(0.9, 0.4, 0.9)
            r, g, b = val.xyz
            out[base + 0] = r
            out[base + 1] = g
            out[base + 2] = b
