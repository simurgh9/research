import numpy as np
# from numba import njit
# from numba.typed import List
from time import perf_counter


def proj(u, v):  # projecting v onto u
    mu = (v @ u.T) / (u @ u.T)
    return mu * u, mu


def gram_schmidt(B):
    U, Mu = np.array(B, dtype=B.dtype), np.zeros(shape=B.shape, dtype=B.dtype)
    for i in range(1, B.shape[1]):
        for j in range(i):
            projection, Mu[i][j] = proj(U[:, j], B[:, i])
            U[:, i] -= projection
    return U, Mu


def lovasz_condition(G, Mu, k, delta):
    c = delta - Mu[k][k - 1]**2
    return G[:, k] @ G[:, k].T >= c * (G[:, k - 1] @ G[:, k - 1].T)


def lll(bad_basis, delta=0.75):
    B = np.array(bad_basis)
    G, Mu = gram_schmidt(B)  # G are the B*
    k, n = 1, B.shape[1] - 1
    while k <= n:
        for j in range(k - 1, -1, -1):
            if abs(Mu[k][j]) > 0.5:  # size condition not satisfied
                B[:, k] -= round(Mu[k][j]) * B[:, j]
                G, Mu = gram_schmidt(B)
        if lovasz_condition(G, Mu, k, delta):
            k = k + 1
        else:
            B[:, [k, k - 1]] = B[:, [k - 1, k]]  # swap
            G, Mu = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


# @njit
def cross(P, N):
    n = len(P)
    squares = set(N)
    for i in range(n):
        v = P[i]
        vn = N[i]
        for j in range(i + 1, n):
            u = P[j]
            un = N[j]
            # t = v - u
            m = np.int64(np.round(np.dot(u, v) / N[j]))
            t = v - (m * u)
            tn = np.dot(t, t)
            if tn not in squares and np.any(t) and (tn < vn or tn < un):
                P.append(t)


def sieve(P, size):
    n, k, start = -1, 0, perf_counter()
    while len(P) != n:
        squares = np.sum(np.square(P), axis=1)
        squares, idx = np.unique(squares, return_index=True)
        N, idx = squares[:size][::-1], idx[:size][::-1]
        P = List([P[i] for i in idx])
        n = len(P)
        print(f'{k:4d} {np.sqrt(N[-1]):7.3f} {np.sqrt(np.mean(N)):7.3f} with {n:5}', end=' ')
        cross(P, N)
        k += 1
        # size = int(0.6 * size)
        print(f'in {perf_counter() - start:3.3f}s')
        start = perf_counter()
    return P


def population(B, size):
    RNG = np.random.default_rng(0)
    p1 = 0.01  # the ratio of 1's in coefs

    t = B.shape[1] * size
    p1 = int(p1 * t)
    coefs = np.array([0] * (t - p1) + [1] * p1)
    RNG.shuffle(coefs)
    coefs = coefs.reshape((B.shape[1], size))

    P = B @ coefs
    P = P[:, ~np.all(P == 0, axis=0)]  # deleting zero columns
    P = np.column_stack((B, P))
    P = np.require(P, dtype=np.float64, requirements=['F'])
    return list(P.T)
