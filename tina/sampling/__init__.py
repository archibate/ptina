from tina.common import *


@ti.func
def wanghash(x):
    value = ti.cast(x, ti.u32)
    value = (value ^ 61) ^ (value >> 16)
    value *= 9
    value ^= value << 4
    value *= 0x27d4eb2d
    value ^= value >> 15
    return int(value)


@ti.func
def wanghash2(x, y):
    value = wanghash(x)
    value = wanghash(y ^ value)
    return value


@ti.func
def wanghash3(x, y, z):
    value = wanghash(x)
    value = wanghash(y ^ value)
    value = wanghash(z ^ value)
    return value


@ti.func
def unixfasthash(x):
    value = ti.cast(x, ti.u32)
    value = (value * 7**5) % (2**31 - 1)
    return int(value)


@ti.pyfunc
def binaryreverse(i):
    j = 0
    k = 1
    while i != 0:
        k <<= 1
        j <<= 1
        j |= i & 1
        i >>= 1
    return j / k


from tina.sampling.sobol import SobolSampler as DefaultSampler
#from tina.sampling.random import RandomSampler as DefaultSampler
