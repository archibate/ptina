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

Camera().set_perspective(np.array([
            [ 1.73205081e+00,  0.00000000e+00,  0.00000000e+00,  1.01348227e-02],
            [ 0.00000000e+00,  1.73205081e+00, -1.73205081e-05, -3.36860025e+00],
            [ 0.00000000e+00, -1.00020002e-05, -1.00020002e+00,  5.27350023e+00],
            [ 0.00000000e+00, -1.00000000e-05, -1.00000000e+00,  5.37243564e+00],
            ]))

nsamples = 32
for i in range(nsamples):
    PathEngine().render()

img = PathEngine().film.get_image()
ti.imshow(img)
