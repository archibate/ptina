from tina.things import *
from tina.engine.path import *
from tina.tools.control import CamControl
from tina.tools.readgltf import readgltf
from tina.tools.readobj import readobj


ti.init(ti.cuda)
init_things()
PathEngine()
FilmTable().set_size(512, 512)

#vertices, mtlids, materials = readgltf('assets/cornell.gltf')
#ModelPool().load(vertices, mtlids)
#MaterialPool().load(materials)

vertices = readobj('assets/monkey.obj')
ModelPool().load(vertices)

BVHTree().build()

gui = ti.GUI()
gui.control = CamControl(gui)
while gui.running:
    if gui.control.process_events():
        FilmTable().clear()
    Camera().set_perspective(gui.control.get_perspective())
    PathEngine().render()
    gui.set_image(ti.imresize(FilmTable().get_image(), 512))
    gui.show()
