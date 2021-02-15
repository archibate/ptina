import numpy as np

f = open('/home/bate/Downloads/new-joe-kuo-6.21201')
f.readline()

# https://web.maths.unsw.edu.au/~fkuo/sobol/
def Sobol(N, D):
    L = int(np.ceil(np.log2(N)))

    C = np.full(N, 0)
    C[0] = 1
    for i in range(N):
        C[i] = 1
        value = i
        while value & 1:
            value >>= 1
            C[i] += 1

    V = np.full((L + 1, D), 0)

    for j in range(D):
        if j != 0:
            _, s, a, M = f.readline().split(maxsplit=3)
            s, a = int(s), int(a)
            m = np.full(s + 1, 0)
            for i, M in enumerate(map(int, M.split())):
                m[i + 1] = M
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

    X = np.full(D, 0)
    for i in range(N):
        P = X / 2**32
        X = X ^ V[C[i], :]
        yield P

for i in Sobol(10, 3):
    print(i)

exit(1)
