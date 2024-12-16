# -*- compile-command: "sage -python hermite.py" -*-
from sage.all import *
import numpy as np


d = 40
RNG = np.random.default_rng(786)
L = RNG.integers(-d**3, d**3, size=(d, d))
# L = RNG.integers(-2**d, 2**d, size=(d, d))
print(L)
A = matrix(L)
A = A.hermite_form().LLL(delta=0.9999999).T

# Challenge Website Threshold
lt = (gamma(d / 2 + 1)**(1/d) / sqrt(pi)) * abs(A.det().n())**(1/d)
lt = ceil(1.05 * lt)

# Gaussian Heuristic (Hoffstein (pg. 402))
gh = ceil(sqrt(d / (2 * pi * e)) * abs(A.det().n())**(1/d))  

print(f'Norm has to be less than: {lt} for {d}')
print(f'Shortest |v| according to GH: {gh}')
out = [','.join([str(e) for e in r]) for r in A]

# with open(f'../../../saved/personal/exp{d}_{lt}_{gh}.hermite.csv', 'w') as fl:
with open(f'../../../saved/personal/col{d}_{lt}_{gh}.hermite.csv', 'w') as fl:
    fl.write('\n'.join(out))
