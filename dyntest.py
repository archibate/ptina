from tina.common import *


class MemoryAllocator:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.free_chunk = [(0, self.size)]
        self.used_chunk = []

    def malloc(self, size):
        for i, (chk_base, chk_size) in enumerate(self.free_chunk):
            if chk_size >= size:
                del self.free_chunk[i]
                if chk_size != size:
                    rest_chunk = (chk_base + size, chk_size - size)
                    self.free_chunk.insert(i, rest_chunk)
                base = chk_base
                break
        else:
            raise RuntimeError('Out of memory!')
        self.used_chunk.append((base, size))
        return base

    def free(self, base):
        for i, (chk_base, chk_size) in enumerate(self.used_chunk):
            if chk_base == base:
                del self.used_chunk[i]
                size = chk_size
                break
        else:
            raise RuntimeError(f'Invalid pointer: {base!r}')

        new_chunk = (base, size)
        self.free_chunk.insert(i, new_chunk)


class IdAllocator:
    def __init__(self, count):
        self.count = count
        self.reset()

    def reset(self):
        self.water = 0

    def malloc(self):
        id = self.water
        if id >= self.count:
            raise RuntimeError('Out of ID!')
        self.water += 1
        return id

    def free(self, id):
        pass


@ti.data_oriented
class MemoryRoot:
    def __init__(self, size, count):
        self.size = size
        self.root = ti.field(int, size)
        self.mman = MemoryAllocator(size)
        self.idman = IdAllocator(count)
        self.sizes = ti.field(int, count)
        self.bases = ti.field(int, count)
        self.shapes = ti.field(int, (count, 8))

    @ti.func
    def subscript(self, index):
        return self.root[index]

    @ti.python_scope
    def new(self, shape):
        size = Vprod(shape)
        base = self.mman.malloc(size)
        id = self.idman.malloc()
        self.bases[id] = base
        self.sizes[id] = size
        for i in range(len(shape)):
            self.shapes[id, i] = shape[i]
        for i in range(len(shape), 8):
            self.shapes[id, i] = 1
        return id

    @ti.python_scope
    def delete(self, id):
        base = self.bases[id]
        self.mman.free(base)
        self.idman.free(id)
        self.bases[id] = 0
        self.sizes[id] = 0
        for i in range(8):
            self.shapes[id, i] = 0

    @ti.func
    def get_view(self, id):
        base = self.bases[id]
        shape = ti.Vector([0] * 8)
        for i in ti.static(range(8)):
            shape[i] = self.shapes[id, i]
        view = MemoryView(self, base, shape)
        return view


@ti.data_oriented
class MemoryView:
    is_taichi_class = True

    def __init__(self, parent, base, shape):
        self.parent = ti.expr_init(parent)
        self.shape = ti.expr_init(shape)
        self.base = ti.expr_init(base)

    @ti.func
    def linearize_indices(self, indices):
        index = self.base
        stride = 1
        for i in ti.static(range(len(indices) - 1, 0, -1)):
            index += stride * (indices[i] % self.shape[i])
            stride *= self.shape[i]
        return index

    def subscript(self, *indices):
        indices = tovector(indices)
        index = self.linearize_indices(indices)
        return self.parent.subscript(index)

    def variable(self):
        return self


mem = MemoryRoot(1024, 32)
id = mem.new((10, 10))

@ti.kernel
def func():
    view = mem.get_view(id)
    for i, j in ti.ndrange(*view.shape.xy):
        view[i, j] = i * 10 + j
    for i, j in ti.ndrange(*view.shape.xy):
        print(i, j, view[i, j])

func()