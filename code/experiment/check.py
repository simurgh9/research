# -*- compile-command: "sage -python check.py" -*-
# https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
from sage.all import *
from sage.modules.free_module_integer import IntegerLattice

with open('challenge.csv', 'r') as fl:
    L = [[int(x) for x in row.split(',')]
         for row in fl.read().split('\n')[:-1]]

n = len(L)
A = matrix(L)
L = IntegerLattice(L)
v = vector(
    (398, 305, 268, -125, -96, -214, -284, 108, -37, 2, -402, -228, 243, 33,
     76, 265, 3, -558, -323, -552, 419, 408, -217, -2, -440, -375, 153, -108,
     -79, -80, 299, 81, -385, 80, 53, 294, 170, -380, -164, -172))
a = [0 for _ in range(40)]
a = vector(a)
a[1] = 22
print(a)
print(a in L)
