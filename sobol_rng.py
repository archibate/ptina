from sobol_lib import *
import timeit

class SobolRNG:
    def __init__(self, skip=32):
        self.skip = 32

    def __call__(self, size, dim=36):
        arr = i4_sobol_generate(dim, size, self.skip)
        self.skip += size
        return arr.transpose()


rng = SobolRNG()
print(timeit.timeit(lambda: rng(128), number=60))
