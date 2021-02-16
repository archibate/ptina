import numpy as np


def load_standard_sobol_file(path):
    print(f'[TinaSobol] reading data from file: {path}')
    with open(path) as f:
        r = []
        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            _, s, a, m = line.split(maxsplit=3)
            s, a = int(s), int(a)
            m = [int(m) for m in m.split()]
            assert len(m) == s, f'{len(m)} != {s}'
            r += [s, a] + m
        return np.array(r)



# https://web.maths.unsw.edu.au/~fkuo/sobol/
def Sobol(N, D):
    from .data import _sobol_data
    file = iter(_sobol_data)

    print(f'[TinaSobol] initializing with N={N}, D={D}')

    L = int(np.ceil(np.log2(N)))

    def C(i):
        bits = 1
        value = i
        while value & 1:
            value >>= 1
            bits += 1
        return bits

    V = np.full((L + 1, D), 0)

    for j in range(D):
        if j != 0:
            s = next(file)
            a = next(file)
            m = np.full(s + 1, 0)
            for i in range(s):
                m[i + 1] = next(file)
        else:
            m = np.full(L + 1, 1)
            s = L

        if L <= s:
            for i in range(L + 1):
                V[i, j] = m[i] << (32 - i)
        else:
            for i in range(s + 1):
                V[i, j] = m[i] << (32 - i)
            for i in range(s + 1, L + 1):
                V[i, j] = V[i - s, j] ^ (V[i - s, j] >> s)
                for k in range(1, s):
                    V[i, j] ^= ((a >> (s - 1 - k)) & 1) * V[i - k, j]

    print('[TinaSobol] start generating sequence')

    X = np.full(D, 0)
    for i in range(N):
        P = X / 2**32
        X = X ^ V[C(i), :]
        yield P
