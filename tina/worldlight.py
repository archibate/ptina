from tina.image import *


@ti.data_oriented
class WorldLight(metaclass=Singleton):
    def __init__(self):
        ...

        '''
        @ti.materialize_callback
        def load_bgm():
            self.bgm = Image.load('assets/env.png')
        '''

    @ti.func
    def at(self, dir):
        if ti.static(hasattr(self, 'bgm')):
            return self.bgm(*dir2tex(dir)).xyz**(1/2.2)
        else:
            return 0.05
