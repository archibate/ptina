from tina.common import *


@ti.func
def schlickFresnel(cost):
    return clamp(1 - cost, 0, 1)**5


@ti.func
def dielectricFresnel(etai, etao, cosi):
    sini = ti.sqrt(max(0, 1 - cosi**2))
    sint = etao / etai * sini

    ret = 1.0
    if sint < 1:
        cost = ti.sqrt(max(0, 1 - sint**2))
        a1, a2 = etai * cosi, etao * cost
        b1, b2 = etao * cosi, etai * cost
        para = (a1 - a2) / (a1 + a2)
        perp = (b1 - b2) / (b1 + b2)
        ret = 0.5 * (para**2 + perp**2)

    return ret


@ti.func
def GTR1(cosh, alpha):
    alpha2 = alpha**2
    t = 1 + (alpha2 - 1) * cosh**2
    return (alpha2 - 1) / (ti.pi * ti.log(alpha2) * t)


@ti.func
def GTR2(cosh, alpha):
    alpha2 = alpha**2
    t = 1 + (alpha2 - 1) * cosh**2
    return alpha2 / (ti.pi * t**2)


@ti.func
def smithGGX(cosi, alpha):
    a = alpha**2
    b = cosi**2
    return 1 / (cosi + ti.sqrt(a + b - a * b))


@ti.func
def sample_GTR1(u, v, alpha):
    u = ti.sqrt(alpha**(2 - 2 * u) - 1) / (alpha**2 - 1)
    return spherical(u, v)


@ti.func
def sample_GTR2(u, v, alpha):
    u = ti.sqrt((1 - u) / (1 - u * (1 - alpha**2)))
    return spherical(u, v)
