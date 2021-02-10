from common import *


class MemoryAllocator:
    def __init__(self, size):
        self.size = size
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


@ti.data_oriented
class Session:
    def __init__(self, size):
        self.size = size
        self.f_mman = MemoryAllocator(self.size)
        self.f_root = ti.field(float, self.size)


_pysession = None

def get_session():
    assert _pysession is not None, 'Session not started, please run init_session()'
    return _pysession


def init_session(size=2**22):
    global _pysession
    _pysession = Session(size)
