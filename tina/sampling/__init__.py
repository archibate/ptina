from tina.common import *
from tina.sampling.sobol import *


@ti.data_oriented
class RNGSobol(metaclass=Singleton):
    def __init__(self):
        self.sobol = TaichiSobol()

    def get_proxy(self, i):
        return self.sobol.get_proxy(i)

    def update(self):
        return self.sobol.update()

    def reset(self):
        return self.sobol.reset()


'''
@ti.data_oriented
class RNGMetropolis(metaclass=Singleton):
    def __init__(self, size=2**18, count=32):
        self.value = ti.field(int, (size, count))
        self.lastid = ti.field(int, (size, count))
        self.len = ti.field(int, size)

    @ti.func
    def calc(self, x, y):
        if y >= self.len[x]:
            self.len[x] = y + 1
            if self.lastid[x, y] < last_large_id:
                self.value[x, y] = ti.random()
                self.lastid[x, y] = last_large_id
            nsmall = curr_id - self.lastid[x, y]
        return self.value[x, y]

    def get_proxy(self, x):
        return self.Proxy(self, x)

    @ti.data_oriented
    class Proxy:
        def __init__(self, parent, x):
            self.parent = parent
            self.x = ti.expr_init(x)
            self.y = ti.expr_init(0)

        @ti.func
        def random(self):
            ret = self.parent.calc(self.x, self.y)
            self.y += 1
            return ret
'''
