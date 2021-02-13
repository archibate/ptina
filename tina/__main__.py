from tina.engine import *
from tina.tools.control import *


ti.init(ti.gpu)
Stack()
Camera()
BVHTree()
ImagePool()
ModelPool()
LightPool()
ToneMapping()
PathEngine()

LightPool().color[0] = V3(4)
LightPool().pos[0] = V(0, 0, 4)
LightPool().radius[0] = 1.0
LightPool().count[None] = 1

ModelPool().load('assets/monkey.obj')
BVHTree().build()

gui = ti.GUI()
gui.control = Control(gui)
while gui.running:
    if gui.control.process_events():
        PathEngine().film.clear()
    Camera().set_perspective(gui.control.get_perspective())
    PathEngine().render()
    gui.set_image(PathEngine().get_image())
    gui.show()
