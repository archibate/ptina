from session import *
from geometries import *


@ti.data_oriented
class BVHTree:
    def __init__(self):
        self.N = N = 2**20
        self.dir = ti.field(int, N)
        self.ind = ti.field(int, N)
        self.min = ti.Vector.field(3, float, N)
        self.max = ti.Vector.field(3, float, N)

    def build(self, pmin, pmax):
        assert len(pmin) == len(pmax)
        assert np.all(pmax >= pmin)
        data = lambda: None
        data.dir = self.dir.to_numpy()
        data.dir[:] = -1
        data.min = self.min.to_numpy()
        data.max = self.max.to_numpy()
        data.ind = self.ind.to_numpy()
        print('[Tina] building tree...')
        self._build(data, pmin, pmax, np.arange(len(pmin)), 1)
        self._build_from_data(data.dir, data.min, data.max, data.ind)
        print('[Tina] building tree done')

    @ti.kernel
    def _build_from_data(self,
            data_dir: ti.ext_arr(),
            data_min: ti.ext_arr(),
            data_max: ti.ext_arr(),
            data_ind: ti.ext_arr()):
        for i in range(self.dir.shape[0]):
            if data_dir[i] == -1:
                continue
            self.dir[i] = data_dir[i]
            for k in ti.static(range(self.dim)):
                self.min[i][k] = data_min[i, k]
                self.max[i][k] = data_max[i, k]
            self.ind[i] = data_ind[i]

    def _build(self, data, pmin, pmax, pind, curr):
        assert curr < self.N_tree, curr
        if not len(pind):
            return

        elif len(pind) <= 1:
            data.dir[curr] = 0
            data.ind[curr] = pind[0]
            data.min[curr] = pmin[0]
            data.max[curr] = pmax[0]
            return

        bmax = np.max(pmax, axis=0)
        bmin = np.min(pmin, axis=0)
        dir = np.argmax(bmax - bmin)
        sort = np.argsort(pmax[:, dir] + pmin[:, dir])
        mid = len(sort) // 2
        lsort = sort[:mid]
        rsort = sort[mid:]

        lmin, rmin = pmin[lsort], pmin[rsort]
        lmax, rmax = pmax[lsort], pmax[rsort]
        lind, rind = pind[lsort], pind[rsort]
        data.dir[curr] = 1 + dir
        data.ind[curr] = 0
        data.min[curr] = bmin
        data.max[curr] = bmax
        self._build(data, lmin, lmax, lind, curr * 2)
        self._build(data, rmin, rmax, rind, curr * 2 + 1)

    @ti.func
    def intersect(self, ray):
        stack = get_stack()
        ntimes = 0
        stack.clear()
        stack.push(1)
        hitind = -1
        hituv = V(0., 0.)
        ret = namespace(hit=0, depth=inf, index=hitind, uv=hituv)

        while ntimes < self.N and stack.size() != 0:
            curr = stack.pop()

            if self.dir[curr] == 0:
                index = self.ind[curr]
                hit = self.geom[index].intersect(ray)
                if hit.hit != 0 and hit.depth < ret.depth:
                    ret.depth = hit.depth
                    ret.index = index
                    ret.uv = hit.uv
                    ret.hit = 1
                continue

            boxhit = Box(self.min[curr], self.max[curr]).intersect(ray)
            if boxhit.hit == 0:
                continue

            ntimes += 1
            stack.push(curr * 2)
            stack.push(curr * 2 + 1)

        return ret
