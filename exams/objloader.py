from ptina.things import *
from ptina.engine.path import *
from ptina.tools.control import CamControl
from ptina.tools.readgltf import readgltf
from ptina.tools.readobj import readobj


ti.init(ti.opengl)
init_things()
PathEngine()
FilmTable().set_size(1024, 1024)

vertices = readobj('assets/monkey.obj')
ModelPool().load(vertices)
BVHTree().build()

gui = ti.GUI('objloader', (1024, 1024))
gui.control = CamControl(gui)
while gui.running:
    if gui.control.process_events():
        FilmTable().clear()
    Camera().set_perspective(gui.control.get_perspective())
    PathEngine().render()
    gui.set_image(FilmTable().get_image())
    gui.show()
