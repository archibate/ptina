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
    def __init__(self, dtype, size, count):
        self.size = size
        self.count = count
        self.root = ti.field(dtype, size)
        self.mman = MemoryAllocator(size)
        self.idman = IdAllocator(count)
        self.sizes = ti.field(int, count)
        self.bases = ti.field(int, count)
        self.shapes = ti.field(int, (count, 8))
        self.args = ti.field(int, 8)
        self.idsi = ti.field(int, 8)

    @ti.func
    def subscript(self, index):
        return self.root[index]

    @ti.python_scope
    def new(self, shape):
        shape = totuple(shape)
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

    @ti.func
    def get_vector_view(self, id, n):
        base = self.bases[id]
        shape = ti.Vector([0] * 8)
        for i in ti.static(range(8)):
            shape[i] = self.shapes[id, i]
        view = MemoryVectorView(self, base, shape, n)
        return view

    @ti.python_scope
    def field(self, shape):
        shape = totuple(shape)
        id = self.new(shape)
        return MemoryField(self, id, shape)

    @ti.python_scope
    def vector_field(self, n, shape):
        shape = totuple(shape) + (n,)
        id = self.new(shape)
        return MemoryVectorField(self, id, shape)


def apply_aug_operation(op, lhs, rhs):
    if op == 'Add':
        return lhs + rhs
    elif op == 'Sub':
        return lhs - rhs
    elif op == 'Mult':
        return lhs * rhs
    elif op == 'Div':
        return lhs / rhs
    elif op == 'FloorDiv':
        return lhs // rhs
    elif op == 'Mod':
        return lhs % rhs
    elif op == 'BitAnd':
        return lhs & rhs
    elif op == 'BitOr':
        return lhs | rhs
    elif op == 'BitXor':
        return lhs ^ rhs
    elif op == 'RShift':
        return lhs >> rhs
    elif op == 'LShift':
        return lhs << rhs
    else:
        assert False, op


@ti.data_oriented
class BitCastField:
    is_taichi_class = True

    @ti.python_scope
    def __init__(self, field, src_type, dst_type):
        self.field = field
        self.src_type = src_type
        self.dst_type = dst_type

    @ti.taichi_scope
    def subscript_assign(self, value, *indices):
        self.field.subscript(*indices).assign(ti.bit_cast(ti.cast(value, self.dst_type), self.src_type))

    @ti.taichi_scope
    def subscript_augassign(self, value, op, *indices):
        self.subscript_assign(apply_aug_operation(op, self.subscript(*indices), value), *indices)

    @ti.taichi_scope
    def subscript(self, *indices):
        ret = ti.bit_cast(self.field.subscript(*indices), self.dst_type)
        def wrapped_assign(value):
            self.subscript_assign(value, *indices)
        def wrapped_augassign(value, op):
            self.subscript_augassign(value, op, *indices)
        ret.assign = wrapped_assign
        ret.augassign = wrapped_augassign
        return ret

    def __getattr__(self, name):
        return getattr(self.field, name)

    @ti.taichi_scope
    def variable(self):
        return self


@ti.data_oriented
class MemoryView:
    is_taichi_class = True

    @ti.taichi_scope
    def __init__(self, parent, base, shape):
        self.parent = parent
        self.shape = ti.expr_init(shape)
        self.base = ti.expr_init(base)

    @ti.func
    def linearize_indices(self, indices):
        index = self.base
        stride = 1
        for i in ti.static(range(len(indices) - 1, -1, -1)):
            index += stride * (indices[i] % self.shape[i])
            stride *= self.shape[i]
        return index

    @ti.taichi_scope
    def subscript(self, *indices):
        indices = tovector(indices)
        index = self.linearize_indices(indices)
        return self.parent.subscript(index)

    @ti.taichi_scope
    def variable(self):
        return self


@ti.data_oriented
class MemoryVectorView(MemoryView):
    @ti.taichi_scope
    def __init__(self, parent, base, shape, n):
        super().__init__(parent, base, shape)
        self.shape = ti.expr_init(ti.Vector(tuple(self.shape.entries[:-1]) + (n,)))
        self.n = n

    @ti.taichi_scope
    def subscript(self, *indices):
        indices = tuple(tovector(indices).entries)
        return ti.Vector([self.parent.subscript(
                self.linearize_indices(ti.Vector(indices + (i,))))
                    for i in range(self.n)])


@ti.data_oriented
class MemoryField:
    is_taichi_class = True

    @ti.python_scope
    def __init__(self, parent, id, shape):
        self.parent = parent
        self.id = id
        self.shape = shape
        self.dim = len(shape)

    @ti.python_scope
    def delete(self):
        self.parent.delete(self.id)

    @property
    @ti.taichi_scope
    def view(self):
        return self.parent.get_view(self.id)

    @ti.taichi_scope
    def subscript(self, *indices):
        assert len(indices) == self.dim, f'{self.dim} indices expected, got {len(indices)}'
        return self.view.subscript(*indices)

    @ti.taichi_scope
    def variable(self):
        return self

    @classmethod
    @ti.kernel
    def _setitem_i(cls: ti.template(), parent: ti.template(), id: int,
            value: int, nindices: ti.template()):
        indices = ti.Vector([0] * nindices)
        for i in ti.static(range(nindices)):
            indices[i] = parent.args[i]
        view = parent.get_view(id)
        view[indices] = value

    @classmethod
    @ti.kernel
    def _getitem_i(cls: ti.template(), parent: ti.template(), id: int,
            nindices: ti.template()) -> int:
        indices = ti.Vector([0] * nindices)
        for i in ti.static(range(nindices)):
            indices[i] = parent.args[i]
        view = parent.get_view(id)
        return view[indices]

    @ti.python_scope
    def __setitem__(self, indices, value):
        indices = totuple(indices)
        for i, v in enumerate(indices):
            self.parent.args[i] = v
        self._setitem_i(self.parent, self.id, value, len(indices))

    @ti.python_scope
    def __getitem__(self, indices):
        indices = totuple(indices)
        for i, v in enumerate(indices):
            self.parent.args[i] = v
        return self._getitem_i(self.parent, self.id, len(indices))


@ti.data_oriented
class MemoryVectorField(MemoryField):
    @ti.python_scope
    def __init__(self, parent, id, shape):
        super().__init__(parent, id, shape)
        self.n = self.shape[-1]
        self.shape = self.shape[:-1]
        self.dim -= 1

    @ti.taichi_scope
    def subscript(self, *indices):
        assert len(indices) == self.dim, f'{self.dim} indices expected, got {len(indices)}'
        return ti.Vector([self.view.subscript(*indices + (i,)) for i in range(self.n)])


g_mem = MemoryRoot(int, 2**16, 32)


@ti.data_oriented
class MObject:
    @ti.python_scope
    def _prepare_defs(self):
        if not hasattr(self, '_parent'):
            self._parent = g_mem
        if not hasattr(self, '_defs'):
            self._defs = []

    @ti.python_scope
    def define(self, name, shape, n=None):
        self._prepare_defs()
        if n is not None:
            shape = totuple(shape) + (n,)
        id = self._parent.new(shape)
        self._defs.append((name, id, n))

    @ti.python_scope
    def _done_defs(self):
        if hasattr(self, '_ids'):
            return
        self._prepare_defs()
        count = len(self._defs)
        self._ids = self._parent.field(count)
        for i, (name, id, n) in enumerate(self._defs):
            self._set_field(i, id)
            proxy = self._FieldProxy(self, name, id, n)
            setattr(self, name, proxy)

    @ti.data_oriented
    class _FieldProxy:
        @ti.python_scope
        def __init__(self, mobject, name, id, n):
            self._mobject = mobject
            self._name = name
            self._id = id
            self._n = n

        @property
        def _core(self):
            if self._n is not None:
                return self._mobject._get_vector_field(self._id, self._n)
            else:
                return self._mobject._get_field(self._id)

        def __getattr__(self, name):
            return getattr(self._core, name)

        def subscript(self, *indices):
            self._core.subscript(*indices)

    @ti.python_scope
    def _set_field(self, i, id):
        self._ids[i] = id

    @ti.python_scope
    def _do_prepare(self, argid):
        self._done_defs()
        idsi = self._ids.id
        self._parent.idsi[argid] = idsi
        self._argid = argid

    @ti.python_scope
    def _do_unprepare(self):
        del self._argid

    @ti.func
    def _get_field_id(self, i):
        idsi = self._parent.idsi[self._argid]
        ids = self._parent.get_view(idsi)
        return ids[i]

    @ti.func
    def _get_field(self, i):
        return self._parent.get_view(self._get_field_id(i))

    @ti.func
    def _get_vector_field(self, i, n):
        return self._parent.get_vector_view(self._get_field_id(i), n)

    def __hash__(self):
        self._prepare_defs()
        return id(type(self)) ^ hash(self._parent)

    def __eq__(self, other):
        if not isinstance(other, MObject):
            return False
        self._prepare_defs()
        other._prepare_defs()
        return type(self) is type(other) and self._parent is other._parent


def _fix_missing_tmpl_anno(func):
    import inspect

    sig = inspect.signature(func)
    for i, (name, param) in enumerate(sig.parameters.items()):
        if i == 0:
            continue
        if param.annotation is inspect.Parameter.empty:
            func.__annotations__[name] = ti.template()


def mokernel(foo):
    def mowrap(func):
        def wrapped(*args, **kwargs):
            exitcbs = []
            for i, obj in enumerate(args):
                if isinstance(obj, MObject):
                    obj._do_prepare(i)
                    exitcbs.append(obj._do_unprepare)

            ret = func(*args, **kwargs)

            [cb() for cb in exitcbs]
            return ret

        return wrapped

    _fix_missing_tmpl_anno(foo)

    from taichi.lang.kernel import _kernel_impl
    foo = _kernel_impl(foo, level_of_class_stackframe=3)

    foo._primal = mowrap(foo._primal)
    foo._adjoint = mowrap(foo._adjoint)

    return foo


class Image(MObject):
    def __init__(self, m, n):
        self.define('img', (m, n), 3)

    @mokernel
    def func(self):
        ti.static_print('jit Image.func')
        print('!!', self.img.shape.xy, self.img.n)


class Texture(Image):
    def __init__(self, m, n):
        self.define('img', (m, n), 2)

    @mokernel
    def func(self):
        ti.static_print('jit Texture.func')
        print('!!', self.img.shape.xy, self.img.n)


class Data(MObject):
    def __init__(self, m, n):
        self.define('dat', (m, n))

    @mokernel
    def func(self, other):
        ti.static_print('jit Data.func')
        print(self.dat.shape.xy)
        print(other.img.shape.xy)
        print('====')


i = Image(3, 4)
j = Image(5, 6)
k = Texture(7, 8)
d = Data(2, 3)
e = Data(1, 2)
d.func(i)
d.func(j)
e.func(j)
d.func(k)
i.func()
j.func()
k.func()
