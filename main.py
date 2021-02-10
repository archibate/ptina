from image import *
from model import *

init_session()

x = Model.load('assets/monkey.obj')
print(x.to_numpy())
