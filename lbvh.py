from tina.common import *


@ti.pyfunc
def expandBits(v):
    v = (v * 0x00010001) & clamp_unsigned(0xFF0000FF)
    v = (v * 0x00000101) & clamp_unsigned(0x0F00F00F)
    v = (v * 0x00000011) & clamp_unsigned(0xC30C30C3)
    v = (v * 0x00000005) & clamp_unsigned(0x49249249)
    return v


@ti.pyfunc
def morton3D(v):
    w = expandBits(clamp(int(v), 0, 1023))
    return w.dot(V(4, 2, 1))


@ti.pyfunc
def clz(x):
    r = 0
    while True:
        f = x >> (31 - r)
        if f == 1 or r == 31:
            r += 1
            break
        r += 1
    return r


@ti.data_oriented
class LinearBVH:
    def __init__(self, n):
        self.n = n

        self.child = ti.Vector.field(2, int, n)
        self.leaf = ti.field(int, n)

        self.mc = ti.field(int, n)
        self.id = ti.field(int, n)


    @ti.func
    def findSplit(self, l, r):
        m = 0

        lc, rc = self.mc[l], self.mc[r]
        if lc == rc:
            m = (l + r) >> 1

        else:
            cp = clz(lc ^ rc)

            m = l
            s = r - l

            while True:
                s += 1
                s >>= 1
                n = m + s

                if n < r:
                    nc = self.mc[n]
                    sp = clz(lc ^ nc)
                    if sp > cp:
                        m = n

                if s <= 1:
                    break

        return m


    @ti.func
    def determineRange(self, n, i):
        l, r = 0, n - 1

        if i != 0:
            ic = self.mc[i]
            lc = self.mc[i - 1]
            rc = self.mc[i + 1]

            if lc == ic == rc:
                l = i
                while i < n - 1:
                    i += 1
                    if i > n - 1:
                        break
                    if self.mc[i] != self.mc[i + 1]:
                        break
                r = i

            else:
                ld = clz(ic ^ lc)
                rd = clz(ic ^ rc)

                d = -1
                if rd > ld:
                    d = 1
                delta_min = min(ld, rd)
                lmax = 2
                delta = -1
                itmp = i * d * lmax
                if 0 < itmp and itmp < n:
                    delta = clz(ic ^ self.mc[itmp])
                while delta > delta_min:
                    lmax <<= 1
                    itmp = i + d * lmax
                    delta = -1
                    if 0 <= itmp < n:
                        delta = clz(ic ^ self.mc[itmp])
                s = 0
                t = lmax >> 1
                while t > 0:
                    itmp = i + (s + t) * d
                    delta = -1
                    if 0 <= itmp and itmp < n:
                        delta = clz(ic ^ self.mc[itmp])
                    if delta > delta_min:
                        s += t
                    t >>= 1

                l, r = i, i + s * d
                if d < 0:
                    l, r = r, l

        return l, r


    @ti.pyfunc
    def get_center(self, i):
        return random3(ti)


    @ti.kernel
    def genMortonCodes(self):
        n = self.n

        for i in range(n):
            center = self.get_center(i)
            self.mc[i] = morton3D(center)
            self.id[i] = i


    def sortMortonCodes(self):
        self.mc_ = self.mc.to_numpy()
        self.id_ = self.id.to_numpy()
        arg = np.argsort(self.mc_)
        self.mc_ = self.mc_[arg]
        self.id_ = self.id_[arg]
        self.mc.from_numpy(self.mc_)
        self.id.from_numpy(self.id_)


    @ti.kernel
    def genHierarchy(self):
        n = self.n

        for i in range(n):
            self.leaf[i] = self.id[i]

        for i in range(n - 1):
            l, r = self.determineRange(n, i)
            split = self.findSplit(l, r)

            lhs = split
            if lhs != l:
                lhs += n

            rhs = split + 1
            if rhs != r:
                rhs += n

            self.child[n + i][0] = lhs
            self.child[n + i][1] = rhs


bvh = LinearBVH(4)
bvh.genMortonCodes()
bvh.sortMortonCodes()
bvh.genHierarchy()

print(bvh.leaf.to_numpy())
print(bvh.child.to_numpy())
exit(1)
