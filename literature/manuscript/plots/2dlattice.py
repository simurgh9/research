import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 300

lowx, highx = -5, 100
lowy, highy = -25, 500
plt.xlim((lowx, highx))
plt.ylim((lowy, highy))
plt.axhline(0, alpha=0.4)
plt.axvline(0, alpha=0.4)

# columns vectors to row vectors to match what I do in sieve.c
bad = np.array([[95, 47],
                [460, 215]], dtype=np.longlong).T
good = np.array([[1, 40],
                 [30, 5]], dtype=np.longlong).T

X = np.arange(start=lowx, stop=highy + 1)  # truncated integers
Y = np.arange(start=lowy, stop=highy + 1)  # truncated integers
C = np.array([[x, y] for x in X for y in Y])

for (B, c, w, lb) in [(good, 'xg', 0.003, 'Good Basis'), (bad, '.r', 0.003, 'Bad Basis')]:
    L = C @ B
    plt.quiver([0, 0], [0, 0],
               B[:, 0],
               B[:, 1],
               width=w,
               color=c[1],
               angles='xy',
               scale_units='xy',
               label=lb,
               scale=1)
    plt.plot(L[:, 0], L[:, 1], 'k.', markersize=3, alpha=1)

plt.tight_layout()
plt.legend()
# plt.show()
plt.savefig('../media/2dlattice.png')
