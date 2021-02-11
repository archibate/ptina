from image import *
from model import *
from engine import *

init_session()

film = Image(512, 512)
geom = Model.load('assets/sphere.obj')
engine = PathEngine(film, geom)

engine.render()
ti.imshow(film.to_numpy_normalized())
