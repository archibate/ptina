from tina.image import *


@ti.data_oriented
class FilmTable(metaclass=Singleton):
    def __init__(self):
        self.channels = ['color']
        for attr in self.channels:
            setattr(self, attr, None)

    @property
    def nx(self):
        return self.color.nx

    @property
    def ny(self):
        return self.color.ny

    def set_size(self, nx, ny):
        for attr in self.channels:
            img = getattr(self, attr)
            if img is not None:
                img.delete()
        for attr in self.channels:
            setattr(self, attr, Image.new(nx, ny))

    def clear(self):
        for attr in self.channels:
            img = getattr(self, attr)
            if img is not None:
                img.clear()

    def get_image(self, raw=False):
        return self.color.get_image(raw=raw)
