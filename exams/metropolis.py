from tina.things import *
from tina.engine.mltpath import *
from tina.tools.control import CamControl
from tina.tools.readgltf import readgltf


ti.init(ti.opengl)
init_things()
#PathEngine()
MLTPathEngine()
FilmTable().set_size(256, 256)

vertices, mtlids, materials = readgltf('assets/caustics.gltf')
ModelPool().load(vertices, mtlids)
MaterialPool().load(materials)
BVHTree().build()

LightPool().pos[0] = [0, 6.7, 0]

gui = ti.GUI()
gui.control = CamControl(gui)
while gui.running:
    if gui.control.process_events():
        FilmTable().clear()
        MLTPathEngine().reset()
    Camera().set_perspective(gui.control.get_perspective())
    MLTPathEngine().render()
    #PathEngine().render()
    img = FilmTable().get_image()
    img = ti.imresize(img**(1/2.2), 512)
    gui.set_image(img)
    gui.show()
