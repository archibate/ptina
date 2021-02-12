from allocator import *


@ti.data_oriented
class ImagePool(metaclass=Singleton):
    is_taichi_class = True

    def __init__(self, size=2**20, count=2**8):
        self.mman = MemoryAllocator(size)
        self.idman = IdAllocator(count)
        self.nx = ti.field(int, count)
        self.ny = ti.field(int, count)
        self.base = ti.field(int, count)
        self.root = ti.Vector.field(4, float, size)

    @ti.func
    def subscript(self, i, x, y):
        index = self.base[i] + x * self.ny[i] + y
        return self.root[index]

    @ti.kernel
    def _to_numpy(self, id: int, arr: ti.ext_arr()):
        nx, ny = self.nx[id], self.ny[id]
        for x, y in ti.ndrange(nx, ny):
            val = self[id, x, y]
            for k in ti.static(range(4)):
                arr[x, y, k] = val[k]

    @ti.kernel
    def _to_numpy_normalized(self, id: int, arr: ti.ext_arr()):
        nx, ny = self.nx[id], self.ny[id]
        for x, y in ti.ndrange(nx, ny):
            val = self[id, x, y]
            if val.w != 0:
                val.xyz /= val.w
            else:
                val.xyz = V(0.9, 0.4, 0.9)
            for k in ti.static(range(3)):
                arr[x, y, k] = val[k]

    def to_numpy(self, id):
        arr = np.empty((self.nx[id], self.ny[id], 4), np.float32)
        self._to_numpy(id, arr)
        return arr

    def to_numpy_normalized(self, id):
        arr = np.empty((self.nx[id], self.ny[id], 3), np.float32)
        self._to_numpy_normalized(id, arr)
        return arr

    @ti.kernel
    def from_numpy(self, id: int, arr: ti.ext_arr()):
        nx, ny = self.nx[id], self.ny[id]
        for x, y in ti.ndrange(nx, ny):
            for k in ti.static(range(4)):
                self[id, x, y][k] = arr[x, y, k]

    def new(self, nx, ny):
        id = self.idman.malloc()
        base = self.mman.malloc(nx * ny)
        self.nx[id] = nx
        self.ny[id] = ny
        self.base[id] = base
        return id

    def delete(self, id):
        base = self.base[id]
        self.idman.free(id)
        self.mman.free(base)

    def load(self, arr):
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

        id = self.new(nx, ny)
        self.from_numpy(id, arr)
        return id


@ti.data_oriented
class Image:
    def __init__(self, id):
        self.id = id

    def to_numpy(self):
        return ImagePool().to_numpy(self.id)

    def to_numpy_normalized(self):
        return ImagePool().to_numpy_normalized(self.id)

    def from_numpy(self, arr):
        return ImagePool().from_numpy(self.id, arr)

    @ti.func
    def subscript(self, x, y):
        x = clamp(x, 0, self.nx)
        y = clamp(y, 0, self.ny)
        return ImagePool()[self.id, x, y]

    @ti.func
    def __call__(self, x, y):
        I = V(x * (self.nx - 1), y * (self.ny - 1))
        return bilerp(self, I)


if __name__ == '__main__':
    ti.init(print_ir=True)
    ImagePool()

    im = Image(ImagePool().load('assets/cloth.jpg'))
    ti.imshow(im.to_numpy())
