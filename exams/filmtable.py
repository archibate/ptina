from tina.things import *
from tina.tools.control import CamControl
from tina.tools.readgltf import readgltf


ti.init(ti.cuda)
init_things()
PathEngine()
FilmTable().set_size(512, 512)

vertices, mtlids, materials = readgltf('assets/cornell.gltf')
ModelPool().load(vertices, mtlids)
MaterialPool().load(materials)
BVHTree().build()

gui = ti.GUI()
gui.control = CamControl(gui)
while gui.running:
    if gui.control.process_events():
        FilmTable().set_size(64, 64)
        FilmTable().clear()
    Camera().set_perspective(gui.control.get_perspective())
    PathEngine().render()
    gui.set_image(ti.imresize(FilmTable().get_image(), 512))
    gui.show()
