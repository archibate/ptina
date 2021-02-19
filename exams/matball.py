from tina.things import *
from tina.engine.brute import *
from tina.tools.control import CamControl
from tina.tools.readgltf import readgltf
from tina.tools.readobj import readobj
from tina.tools.globals import *


ti.init(ti.cuda)
init_things()
Globals().add('metallic', 1, 0, 1)
Globals().add('roughness', 0, 0, 1)

BruteEngine()
FilmTable().set_size(256, 256)

vertices = readobj('assets/sphere.obj')
ModelPool().load(vertices)

BVHTree().build()

gui = ti.GUI()
gui.control = CamControl(gui)
while gui.running:
    if gui.control.process_events():
        FilmTable().clear()
    Camera().set_perspective(gui.control.get_perspective())
    Globals().update(gui)
    BruteEngine().render()
    img = ti.imresize(FilmTable().get_image(), 512)
    gui.set_image(img)
    gui.show()
