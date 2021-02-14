from tina.image import *


@ti.data_oriented
class ParameterPair:
    def __init__(self, count):
        self.fac = ti.Vector.field(3, float, count)
        self.tex = ti.field(int, count)

    def load(self, i, fac, tex):
        if fac is None:
            fac = 1.0
        if not isinstance(fac, (tuple, list)):
            fac = [fac, fac, fac]
        self.fac[i] = fac
        if tex is None:
            self.tex[i] = -1
        else:
            imgid = ImagePool().load(tex)
            self.tex[i] = imgid
    
    @ti.func
    def get(self, mtlid, texcoord):
        fac = self.fac[mtlid]
        texid = self.tex[mtlid]
        tex = Image(texid)
        return fac * tex(*texcoord)


@ti.data_oriented
class MaterialPool(metaclass=Singleton):
    def __init__(self, count=2**6):
        self.basecolor = ParameterPair(count)
        self.metallic = ParameterPair(count)
        self.roughness = ParameterPair(count)
        self.count = ti.field(int, ())

    def load(self, materials):
        for i, material in enumerate(materials):
            b, bt, m, mt, r, rt = material
            self.basecolor.load(b, bt)
            self.metallic.load(m, mt)
            self.roughness.load(r, rt)

    def get(self, mtlid, texcoord):
        material = Disney(
                self.basecolor.get(mtlid, texcoord),
                self.metallic.get(mtlid, texcoord),
                self.roughness.get(mtlid, texcoord),
                )
        return material
