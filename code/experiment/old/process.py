# -*- compile-command: "sage -python process.py" -*-
# https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
from sage.all import *

PATH = '../../../saved/bases/'

for dim in range(40, 200, 2):
    with open(f'{PATH}challenge{dim}.csv', 'r') as fl:
        L = [[int(x) for x in row.split(',')] for row in fl.read().split('\n')[:-1]]

    n = len(L)
    A = matrix(L)
    lt = ceil(1.05 * ((gamma(n / 2 + 1)**(1/n)) / sqrt(pi)) * (A.det().n().nth_root(n)))
    print(f'Norm has to be less than: {lt} for {n}')
    A = A.LLL(delta=0.9999999).T
    # # A = A.BKZ(delta=0.9999999, block_size=60, proof=False).T
    out = [','.join([str(e) for e in r]) for r in A]

    with open(f'{PATH}col{dim}_{lt}.csv', 'w') as fl:
        fl.write('\n'.join(out))
