from session import *


@ti.data_oriented
class Model:
    is_taichi_class = True

    def __init__(self, nverts):
        self.nverts = nverts
        self.sess = get_session()
        self.base = self.sess.f_mman.malloc(nverts)

    @ti.func
    def subscript(self, i):
        index = self.base + i * 8
        return tovector([self.sess.f_root[index + i] for i in range(8)])

    @ti.kernel
    def _to_numpy(self, arr: ti.ext_arr()):
        for i in range(self.nverts):
            for k in ti.static(range(8)):
                arr[i, k] = self[i][k]

    def to_numpy(self):
        arr = np.empty((self.nverts, 8), dtype=np.float32)
        self._to_numpy(arr)
        return arr

    @ti.kernel
    def from_numpy(self, arr: ti.ext_arr()):
        for i in range(self.nverts):
            for k in ti.static(range(8)):
                self[i][k] = arr[i, k]

    @classmethod
    def load(cls, arr):
        if isinstance(arr, str):
            from tools.readobj import readobj
            arr = readobj(arr)

        if isinstance(arr, dict):
            verts = arr['v'][arr['f'][:, :, 0]]
            norms = arr['vn'][arr['f'][:, :, 2]]
            coors = arr['vt'][arr['f'][:, :, 1]]
            verts = verts.reshape(arr['f'].shape[0] * 3, 3)
            norms = norms.reshape(arr['f'].shape[0] * 3, 3)
            coors = coors.reshape(arr['f'].shape[0] * 3, 2)
            arr = np.concatenate([verts, norms, coors], axis=1)

        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        nverts = arr.shape[0]
        self = cls(nverts)

        self.from_numpy(arr)

        return self
