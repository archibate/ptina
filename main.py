from image import *
from model import *
from engine import *

init_session()

film = Image(512, 512)
engine = PathEngine(film)

engine.render()
ti.imshow(film.to_numpy_normalized())
