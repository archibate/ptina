import numpy as np
import timeit
import numba
from sobol_data import get_poly, get_vgrid


POLY = get_poly()
V_TMPL = get_vgrid()
DIM, LOG = V_TMPL.shape[:2]
print(DIM, LOG)


N1 = 0xfffffffe


def GetHighestBitPos(n):
    # Returns the position of the high 1 bit base 2 in an integer.
    bit = 0
    while n > 0:
        bit += 1
        n >>= 1
    return bit


def GetLowestBitPos(n):
    # Returns the position of the low 0 bit base 2 in an integer.
    bit = 1
    while n != (n & N1):
        bit += 1
        n >>= 1
    return bit


def GenVG(dim):
    vg = np.full((dim, LOG), 0)

    for i in range(dim):
        for j in range(LOG):
            vg[i, j] = V_TMPL[i, j]

    for i in range(2, dim + 1):
        m = GetHighestBitPos(POLY[i - 1] >> 1)

        inc = np.full(m, 0)

        l = POLY[i - 1]
        for k in range(m, 0, -1):
            inc[k - 1] = l != (l & N1)
            l >>= 1

        for j in range(m + 1, LOG + 1):
            newv = vg[i - 1, j - m - 1]
            l = 1
            for k in range(1, m + 1):
                l <<= 1
                if inc[k - 1]:
                    newv ^= l * vg[i - 1, j - k - 1]
            vg[i - 1, j - 1] = newv

    l = 1
    for j in range(LOG, 0, -1):
        for k in range(dim):
            vg[k, j - 1] *= l
        l <<= 1

    return vg


class Sobol:
    def __init__(self, dim, nsamples, skip):
        self.dim = dim
        self.nsamples = nsamples
        self.vg = GenVG(self.dim)
        self.seed = 0
        self.last_q = np.full(self.dim, 0)
        self.last_sample = np.full(self.dim, -1)
        self.nshift = GetHighestBitPos(nsamples)
        assert self.nshift != 0

        for i in range(skip):
            self.nextImpl()

        assert skip == self.seed

    def nextImpl(self):
        l = GetLowestBitPos(self.seed)
        if LOG <= l:
            self.seed = 0
            l = 0
        else:
            self.seed += 1

        self.last_sample = self.last_q >> (LOG + 1 - self.nshift)
        self.last_q ^= self.vg[:, l]

        return np.any(self.nsamples < self.last_sample)

    def next(self):
        while self.nextImpl():
            print('invalid sample encountered!')
        return self.last_sample


n = 1024
rng = Sobol(1111, 1024, 173)
