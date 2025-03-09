# -*- compile-command: "sage -python intlat.py" -*-
from sage.all import *
import numpy as np


RNG = np.random.default_rng(786)
for d in range(40, 210, 10):
    L = RNG.integers(-d**3, d**3, size=(d, d))
    A = matrix(L).hermite_form().LLL(delta=0.9999999)
    # Challenge Website Threshold
    lt = (gamma(d / 2 + 1)**(1/d) / sqrt(pi)) * abs(A.det().n())**(1/d)
    lt = ceil(1.05 * lt)
    # Gaussian Heuristic (Hoffstein (pg. 402))
    gh = ceil(sqrt(d / (2 * pi * e)) * abs(A.det().n())**(1/d))

    A = A.T  # because we want column vectors
    out = [','.join([str(e) for e in r]) for r in A]

    filename = f'paper{d}_{lt}_{gh}.csv'
    with open(f'../../../saved/paper/int/{filename}', 'w') as fl:
        fl.write('\n'.join(out))
