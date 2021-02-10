from image import *


@ti.data_oriented
class Texture(Image):
    @ti.func
    def subscript(self, x, y):
        x = clamp(x, 0, self.nx)
        y = clamp(y, 0, self.ny)
        return Image.subscript(self, x, y)

    @ti.func
    def __call__(self, x, y):
        I = V(x * (self.nx - 1), y * (self.ny - 1))
        return bilerp(self, I)
