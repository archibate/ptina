from tina.stack import *
from tina.camera import *
from tina.tree import *
from tina.image import *
from tina.model import *
from tina.light import *
from tina.light.world import *
from tina.mtllib import *
from tina.filmtable import *


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
