from session import *


@ti.data_oriented
class ImagePool:
    is_taichi_class = True

    def __init__(self, size=2**22, count=2**8):
        self.mman = MemoryAllocator(size)
        self.idman = IdAllocator(count)
        self.meta = ti.field(int, count * 3)
        self.root = ti.field(float, size)

    @ti.pyfunc
    def nx(self, i):
        return self.meta[i * 3 + 0]

    @ti.pyfunc
    def ny(self, i):
        return self.meta[i * 3 + 1]

    @ti.pyfunc
    def base(self, i):
        return self.meta[i * 3 + 2]

    @ti.pyfunc
    def set_nx(self, i, val):
        self.meta[i * 3 + 0] = val

    @ti.pyfunc
    def set_ny(self, i, val):
        self.meta[i * 3 + 1] = val

    @ti.pyfunc
    def set_base(self, i, val):
        self.meta[i * 3 + 2] = val

    @ti.func
    def subscript(self, i, x, y):
        index = self.base(i) + (x * self.ny(i) + y) * 4
        return tovector([self.root[index + j] for j in range(4)])

    @ti.kernel
    def do_to_numpy(self, idx: int, arr: ti.ext_arr()):
        for x, y in ti.ndrange(self.nx(idx), self.ny(idx)):
            val = self[idx, x, y]
            for k in ti.static(range(4)):
                arr[x, y, k] = val[k]

    @ti.kernel
    def do_to_numpy_normalized(self, idx: int, arr: ti.ext_arr()):
        for x, y in ti.ndrange(self.nx(idx), self.ny(idx)):
            val = self[idx, x, y]
            if val.w != 0:
                val.xyz /= val.w
            else:
                val.xyz = V(0.9, 0.4, 0.9)
            for k in ti.static(range(3)):
                arr[x, y, k] = val[k]

    @ti.kernel
    def do_from_numpy(self, idx: int, arr: ti.ext_arr()):
        for x, y in ti.ndrange(self.nx(idx), self.ny(idx)):
            for k in ti.static(range(4)):
                pass#self[idx, x, y][k] = arr[x, y, k]


@ti.data_oriented
class Image:
    is_taichi_class = True

    def __init__(self, nx, ny):
        self.idx = pool.idman.malloc()
        self.base = pool.mman.malloc(nx * ny * 4)
        self.nx = nx
        self.ny = ny

    @property
    def nx(self):
        return pool.nx(self.idx)

    @nx.setter
    def nx(self, val):
        pool.set_nx(self.idx, val)

    @property
    def ny(self):
        return pool.ny(self.idx)

    @ny.setter
    def ny(self, val):
        pool.set_ny(self.idx, val)

    @property
    def base(self):
        return pool.base(self.idx)

    @base.setter
    def base(self, val):
        pool.set_base(self.idx, val)

    def close(self):
        pool.idman.free(self.idx)
        pool.mman.free(self.base)

    @ti.func
    def subscript(self, x, y):
        return pool.subscript(self.idx, x, y)

    def to_numpy(self):
        arr = np.empty((self.nx, self.ny, 4), dtype=np.float32)
        pool.do_to_numpy(self.idx, arr)
        return arr

    def to_numpy_normalized(self):
        arr = np.empty((self.nx, self.ny, 3), dtype=np.float32)
        pool.do_to_numpy_normalized(self.idx, arr)
        return arr

    def from_numpy(self, arr):
        pool.do_from_numpy(self.idx, arr)

    @classmethod
    def load(cls, arr):
        if isinstance(arr, str):
            arr = ti.imread(arr)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255

        nx, ny = arr.shape[0], arr.shape[1]
        if len(arr.shape) == 2:
            arr = arr[:, :, None]
        if arr.shape[2] == 1:
            arr = np.stack([arr[:, :, 0]] * 3, axis=2)
        if arr.shape[2] == 3:
            arr = np.concatenate([arr, np.ones((nx, ny, 1))], axis=2)
        self = cls(nx, ny)

        self.from_numpy(arr)

        return self


if __name__ == '__main__':
    ti.init(print_ir=True)
    pool = ImagePool()
    im = Image.load('assets/cloth.jpg')
    ti.imshow(im.to_numpy())
