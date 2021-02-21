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


@ti.pyfunc
def findSplit(l, r):
    m = 0

    lc, rc = mc[l], mc[r]
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
                nc = mc[n]
                sp = clz(lc ^ nc)
                if sp > cp:
                    m = n

            if s <= 1:
                break

    return m


@ti.pyfunc
def determineRange(n, i):
    l, r = 0, n - 1

    if i != 0:
        ic = mc[i]
        lc = mc[i - 1]
        rc = mc[i + 1]

        if lc == ic == rc:
            l = i
            while i < n - 1:
                i += 1
                if i > n - 1:
                    break
                if mc[i] != mc[i + 1]:
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
                delta = clz(ic ^ mc[itmp])
            while delta > delta_min:
                lmax <<= 1
                itmp = i + d * lmax
                delta = -1
                if 0 <= itmp < n:
                    delta = clz(ic ^ mc[itmp])
            s = 0
            t = lmax >> 1
            while t > 0:
                itmp = i + (s + t) * d
                delta = -1
                if 0 <= itmp and itmp < n:
                    delta = clz(ic ^ mc[itmp])
                if delta > delta_min:
                    s += t
                t >>= 1

            l, r = i, i + s * d
            if d < 0:
                l, r = r, l

    return l, r


n = 4
child = ti.Vector.field(2, int, n - 1)  # [0] for childA, [1] for childB
leaf = ti.field(int, n)                 # primitive ids for leaf nodes

center = ti.Vector.field(3, float, n)   # input primitive center coordinates
mc = ti.field(int, n)                   # 3d morton codes for primitives
id = ti.field(int, n)                   # primitive ids of corr. sorted mc


@ti.kernel
def genMortonCodes():
    for i in range(n):
        mc[i] = morton3D(center[i])
        id[i] = i


def sortMortonCodes():
    mc_ = mc.to_numpy()
    id_ = id.to_numpy()
    arg = np.argsort(mc_)
    mc_ = mc_[arg]
    id_ = id_[arg]
    mc.from_numpy(mc_)
    id.from_numpy(id_)


@ti.kernel
def genHierarchy():
    for i in range(n):
        leaf[i] = id[i]

    for i in range(n - 1):
        l, r = determineRange(n, i)
        split = findSplit(l, r)

        lhs = split
        if lhs != l:
            lhs += n  # move from leaf -> internal

        rhs = split + 1
        if rhs != r:
            rhs += n  # move from leaf -> internal

        child[n + i][0] = lhs
        child[n + i][1] = rhs


center.from_numpy(np.array([
    [120, 480, 256],
    [640, 512, 64],
    [455, 512, 32],
    [256, 12, 768],
    ]))
genMortonCodes()
sortMortonCodes()
genHierarchy()

print(leaf.to_numpy())
print(child.to_numpy())
exit(1)
