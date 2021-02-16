from tina.things import *
from tina.tools.control import Control
from tina.tools.readgltf import readgltf


ti.init(ti.cuda)
init_things()
PathEngine()

vertices, mtlids, materials = readgltf('assets/cornell.gltf')
ModelPool().load(vertices, mtlids)
MaterialPool().load(materials)
BVHTree().build()

gui = ti.GUI()
gui.control = Control(gui)
while gui.running:
    if gui.control.process_events():
        PathEngine().film.clear()
    Camera().set_perspective(gui.control.get_perspective())
    PathEngine().render()
    gui.set_image(PathEngine().film.get_image())
    gui.show()
