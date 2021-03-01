from ptina.stack import *
from ptina.camera import *
from ptina.tree import *
from ptina.image import *
from ptina.model import *
from ptina.light import *
from ptina.light.world import *
from ptina.mtllib import *
from ptina.filmtable import *


def init_things(
    max_faces=2**21,
    max_texels=2**22,
    max_materials=2**6,
    max_textures=2**6,
    max_lights=2**6,
    max_filmsize=2**21,
    max_filmpasses=3):
    Stack()
    Camera()
    BVHTree(max_faces)
    ImagePool(max_texels, max_textures)
    ModelPool(max_faces)
    LightPool(max_lights)
    WorldLight()
    MaterialPool(max_materials)
    FilmTable(max_filmsize, max_filmpasses)
