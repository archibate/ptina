from session import *


@ti.data_oriented
class Image:
    is_taichi_class = True

    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.sess = get_session()
        self.base = self.sess.f_mman.malloc(nx * ny * 4)

    def __del__(self):
        self.sess.f_mman.free(self.base)

    @ti.func
    def subscript(self, x, y):
        index = self.base + (x * self.ny + y) * 4
        return tovector([self.sess.f_root[index + i] for i in range(4)])

    @ti.kernel
    def _to_numpy(self, arr: ti.ext_arr()):
        for x, y in ti.ndrange(self.nx, self.ny):
            val = self[x, y]
            for k in ti.static(range(4)):
                arr[x, y, k] = val[k]

    @ti.kernel
    def _to_numpy_normalized(self, arr: ti.ext_arr()):
        for x, y in ti.ndrange(self.nx, self.ny):
            val = self[x, y]
            if val.w != 0:
                val.xyz /= val.w
            else:
                val.xyz = V(0.9, 0.4, 0.9)
            for k in ti.static(range(3)):
                arr[x, y, k] = val[k]

    def to_numpy(self):
        arr = np.empty((self.nx, self.ny, 4), dtype=np.float32)
        self._to_numpy(arr)
        return arr

    def to_numpy_normalized(self):
        arr = np.empty((self.nx, self.ny, 3), dtype=np.float32)
        self._to_numpy_normalized(arr)
        return arr

    @ti.kernel
    def from_numpy(self, arr: ti.ext_arr()):
        for x, y in ti.ndrange(self.nx, self.ny):
            for k in ti.static(range(4)):
                self[x, y][k] = arr[x, y, k]

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
