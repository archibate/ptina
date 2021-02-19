from tina.things import *
from tina.engine.mltpath import *
from tina.tools.control import CamControl
from tina.tools.readgltf import readgltf


ti.init(ti.cuda)
init_things()
MLTPathEngine()
FilmTable().set_size(512, 512)

vertices, mtlids, materials = readgltf('assets/cornell.gltf')
ModelPool().load(vertices, mtlids)
MaterialPool().load(materials)
BVHTree().build()

gui = ti.GUI()
gui.control = CamControl(gui)
while gui.running:
    if gui.control.process_events():
        FilmTable().clear()
        MLTEngine().reset()
    Camera().set_perspective(gui.control.get_perspective())
    MLTEngine().render()
    gui.set_image(FilmTable().get_image())
    gui.show()
