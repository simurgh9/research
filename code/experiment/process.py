# -*- compile-command: "sage -python process.py" -*-
# https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
from sage.all import *

with open('challenge.csv', 'r') as fl:
    L = [[int(x) for x in row.split(',')] for row in fl.read().split('\n')[:-1]]

n = len(L)
A = matrix(L)
print('Norm has to be less than:', end=' ')
print(float(1.05 * ((gamma(n / 2 + 1)**(1/n)) / sqrt(pi)) * (A.det()**(1/n))))

# A = A.LLL(delta=0.9999999).T
A = A.BKZ(delta=0.9999999, block_size=60, proof=False).T
out = [','.join([str(e) for e in r]) for r in A]

with open('columns.csv', 'w') as fl:
    fl.write('\n'.join(out))
