# -*- compile-command: "sage -python random_lattice.py" -*-
# https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
from sage.all import *
import numpy as np


d = 70
seeds = {40: 1, 50: 1, 60: 1, 70: 0}
RNG = np.random.default_rng(seeds[d])
L = RNG.integers(-d**3, d**3, size=(d, d))
A = matrix(L)
lt = (gamma(d / 2 + 1)**(1/d) / sqrt(pi)) * A.det().n()**(1/d)
lt = ceil(1.05 * lt)
# Gaussian Heuristic (Hoffstein (pg. 402))
gh = ceil(sqrt(d / (2 * pi * e)) * A.det().n()**(1/d))  
print(f'Norm has to be less than: {lt} for {d}')
print(f'Shortest |v| according to GH: {gh}')
# A = A.LLL(delta=0.9999999).T
# A = A.BKZ(delta=0.9999999, block_size=60, proof=False).T
out = [','.join([str(e) for e in r]) for r in A]

with open(f'../../../saved/personal/col{d}_{lt}_{gh}.csv', 'w') as fl:
    fl.write('\n'.join(out))
