from tina.things import *
#from tina.engine.mltpath import MLTPathEngine as DefaultEngine
from tina.engine.path import PathEngine as DefaultEngine


def init():
    ti.init(ti.cuda)
    init_things()
    DefaultEngine()


def render(aa=True):
    DefaultEngine().render()


def set_size(nx, ny):
    FilmTable().set_size(nx, ny)


def get_size():
    return FilmTable().nx, FilmTable().ny


def clear():
    if hasattr(DefaultEngine(), 'reset'):
        DefaultEngine().reset()
    FilmTable().clear(id)


def get_image(id=0):
    return FilmTable().get_image(id)


def fast_export_image(pixels, id=0):
    FilmTable().fast_export_image(pixels, id)


def clear_lights():
    LightPool().clear()


def add_light(world, color, size, type):
    LightPool().add(world, color, size, type)


def load_model(vertices, mtlids):
    ModelPool().load(vertices, mtlids)


def build_tree():
    BVHTree().build()


def set_camera(pers):
    Camera().set_perspective(pers)
