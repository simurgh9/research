import numpy as np
from lib.lattice import lll
from matplotlib import pyplot as plt

# plt.style.use('dark_background')
plt.figure(figsize=(10, 10))
norm = np.linalg.norm


def set_axis_size(B, pad=0.3):
    M = np.ceil((1 + pad) * np.max(np.abs(B), axis=1))
    plt.xlim([-M[0], M[0]])
    plt.ylim([-M[1], M[1]])
    # plt.axhline(0, alpha=0.4)
    # plt.axvline(0, alpha=0.4)


def plot_vector(v, origin=[0, 0], c='black', a=1):
    return plt.quiver(origin[0],
                      origin[1],
                      v[0] - origin[0],
                      v[1] - origin[1],
                      color=c,
                      alpha=a,
                      width=0.0025,
                      angles='xy',
                      scale_units='xy',
                      scale=1)


def plot_lattice(B, pad=0.3):  # columns are vectors
    M = np.ceil((1 + pad) * np.max(np.abs(B), axis=1))
    Z = [np.arange(-M[0], M[0]), np.arange(-M[1], M[1])]
    X = np.array([[x, y] for x in Z[0] for y in Z[1]]).T
    L = B @ X
    plt.plot(L[0], L[1], 'ok', markersize=3, alpha=0.6)
    set_axis_size(B, pad)


def plot_basis(B):
    # set_axis_size(B)
    a = plot_vector(B[:, 0], c='green')
    b = plot_vector(B[:, 1], c='red')
    plt.draw()
    plt.pause(1)
    a.remove()
    b.remove()


def gaussian_reduction(B):
    while True:
        plot_basis(B)
        if norm(B[:, 1]) < norm(B[:, 0]):
            B[:, [0, 1]] = B[:, [1, 0]]
            continue  # just so we replot
        m = (B[:, 0] @ B[:, 1]) / (B[:, 0] @ B[:, 0])
        if np.round(m) == 0:
            break
        B[:, 1] -= np.round(m).astype(B.dtype) * B[:, 0]


def sieve(P):
    erase, n = [], -1
    while len(P) != n:
        norm_square = np.sum(np.square(P), axis=1)
        norm_square, idx = np.unique(norm_square, return_index=True)
        idx = idx[:size][::-1]
        N = norm_square[:size][::-1]
        norm_square = set(norm_square)
        P = [P[i] for i in idx]
        n = len(P)
        erase = [v.remove() for v in erase]
        erase = [plot_vector(v, c='red', a=0.5) for v in P]
        for i in range(n):
            v, vn = P[i], N[i]
            for j in range(i + 1, n):
                u, un = P[j], N[j]
                t = v - u
                # m = np.round((u @ v) / N[j]).astype(np.int64)
                # t = v - (m * u)
                tn = t @ t.T
                if tn not in norm_square and np.any(t) and (tn < vn or tn < un):
                    P.append(t)
                    erase.append(plot_vector(v, u, c='green'))
        plt.title(f'Size: {n}'), plt.draw(), plt.pause(1)
    return P


if __name__ == "__main__":
    B = np.array([[3, 0], [0, 7]])
    plot_lattice(B, pad=6)
    # B = np.array([[95, 47], [460, 215]])
    # plot_lattice(B, pad=1)
    # B = np.array([[6513996, 66586820], [6393464, 65354729]])
    # plot_lattice(B)

    # visualise gaussian reduction
    # gaussian_reduction(B)

    # visualise sieving
    RNG = np.random.default_rng(0)
    size = 5
    coefs = RNG.normal(scale=3, size=(B.shape[1], size)).astype(np.int64)
    P = (B @ coefs)
    P = P[:, ~np.all(P == 0, axis=0)]  # deleting zero columns
    P = list(P.T)
    sieve(P)
