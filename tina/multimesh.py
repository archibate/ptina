import numpy as np


def compose_multiple_meshes(primitives):
    def np34(x, w):
        return np.concatenate([x, np.full((x.shape[0], 1), w)], axis=1)

    def np43(x):
        return x[:, :3] / np.repeat(x[:, 3, None], 3, axis=1)

    def npnmlz(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def transform_primitive(p, n, t, w, m):
        assert w is not None
        assert p is not None
        assert n is not None

        if t is None:
            t = np.zeros((p.shape[0], 2))

        if m is None:
            m = -1

        p = p.reshape(p.shape[0] * 3, 3)
        n = n.reshape(n.shape[0] * 3, 3)
        t = t.reshape(t.shape[0] * 3, 2)

        assert p.shape[0] == n.shape[0] == t.shape[0]

        p = p.astype(np.float64)
        n = n.astype(np.float64)
        t = t.astype(np.float64)
        w = w.astype(np.float64)

        w = w.transpose()
        p = np43(np34(p, 1) @ w)
        n = npnmlz((np34(n, 0) @ w)[:, :3])

        a = np.concatenate([p, n, t], axis=1)
        assert a.shape[0] % 3 == 0
        m = np.full(a.shape[0] // 3, m)
        return a, m

    vertices = []
    mtlids = []
    for p, n, t, w, m in primitives:
        a, m = transform_primitive(p, n, t, w, m)
        vertices.append(a)
        mtlids.append(m)
    assert len(vertices) and len(mtlids)

    vertices = np.concatenate(vertices, axis=0)
    mtlids = np.concatenate(mtlids, axis=0)

    assert len(vertices) == len(mtlids) * 3

    print('[Tina] loaded', len(mtlids), 'triangles')

    return vertices, mtlids
