import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 300

def set_axis_size(B, pad=0.3):
    M = np.ceil((1 + pad) * np.max(np.abs(B), axis=1))
    plt.xlim([-M[0], M[0]])
    plt.ylim([-M[1], M[1]])


def plot_vector(v, origin=[0, 0], c='black', a=1):
    return plt.quiver(origin[0],
                      origin[1],
                      v[0] - origin[0],
                      v[1] - origin[1],
                      color=c,
                      alpha=a,
                      label=', '.join([str(x) for x in v]),
                      width=0.003,
                      angles='xy',
                      scale_units='xy',
                      scale=1)


def plot_lattice(B, pad=0.3):  # columns are vectors
    M = np.ceil((1 + pad) * np.max(np.abs(B), axis=1))
    Z = [np.arange(-M[0], M[0]), np.arange(-M[1], M[1])]
    X = np.array([[x, y] for x in Z[0] for y in Z[1]]).T
    L = B @ X
    plt.plot(L[0], L[1], 'ok', markersize=3, alpha=0.4)
    set_axis_size(B, pad)


def cartesian_product(P):
    cart = []
    for i in range(P.shape[1]):
        for j in range(i + 1, P.shape[1]):
            cart.append([P[:, i], P[:, j]])
    return cart


def select(P, n):
    l2 = L2(P, axis=0)
    idx = np.argsort(l2)
    P = P[:, idx]
    _, idx = np.unique(l2[idx], return_index=True)
    return P[:, idx][:,:n]


def plot_population(P, title=''):
    colours = list('bgrcmy')
    for i in range(1, P.shape[1]+1):
        plot_vector(P[:, -i], c=colours[i % len(colours)], a=0.7)
    plt.axhline(0, alpha=0.4)
    plt.axvline(0, alpha=0.4)
    set_axis_size(P, pad=0.1)
    plt.title(title)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    # plt.savefig(f'../media/naive{title[-1]}.png')
    plt.show()


L2 = np.linalg.norm
B = np.array([[95, 47], [460, 215]], dtype=np.int64)
size = 4
RNG = np.random.default_rng(5)
coefs = RNG.normal(scale=size, size=(B.shape[1], size)).astype(np.int64)
P = -1 * (B @ coefs)  # makes them all positive
P = P[:, ~np.all(P == 0, axis=0)]  # deleting zero columns
new = np.empty(shape=(2, 0), dtype=np.int64)
i = 0

while True:
    P = select(np.hstack((P, new)), P.shape[1])
    i += 1
    # print(P)
    # plot_lattice(B)
    # plot_population(P, f'Iteration {i}')
    new = np.empty(shape=(2, 0), dtype=np.int64)
    P2 = cartesian_product(P)
    for u, v in P2:
        t = u - v
        if L2(t) not in L2(P, axis=0) and (0 < L2(t) < L2(u) or 0 < L2(t) < L2(v)):
            new = np.hstack((new, t.reshape(-1, 1)))
    if new.shape[1] == 0:
        break
