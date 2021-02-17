from tina.things import *
from threading import Thread, Queue


def init():
    ti.init(ti.cuda)
    init_things()
    PathEngine()


def render(aa=True):
    PathEngine().render(aa)


def set_size(nx, ny):
    FilmTable().set_size(nx, ny)


def get_size():
    return FilmTable().nx, FilmTable().ny


def clear(id=0):
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
