# -*- compile-command: "sage -python process.py" -*-
# https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
from sage.all import *

"""
Note on Matrix Format

https://www.latticechallenge.org/svp-challenge/download/format.txt

The above shows that the lattice basis given on the SVP Challenges
page formats their lattice basis row-vise. I. e., a basis with basis
vectors v = [1, 2] and u = [3, 4] will be written as the matrix,

[1 2]
[3 4]

This is the same format sage follows for it's matrices per note under,

https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/special.html#sage.matrix.special.column_matrix

> Linear algebra in Sage favors rows over columns.

Therefore we read row vectors from the challenge files into a row
matrix, which is compatibly passed to the `matrix(...)` function and
ran under LLL.

The results are then transposed into column vectors before being
written to be read by Tashfeen's C experiment.  The reason is that,
Tashfeen's C experiment reads columns into rows with:

    void basis_t(char [], num (*)[], norm *);

Therefore, the rows transposed into columns get read from columns back
into rows.
"""


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
