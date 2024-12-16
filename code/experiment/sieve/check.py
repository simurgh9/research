# -*- compile-command: "sage -python check.py" -*-
# https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix_integer_dense.html
from sage.all import *
from sage.modules.free_module_integer import IntegerLattice


PATH = '../../saved/personal/col70_2160237_1979552.hermite.csv'
PATH = '../' + PATH

with open(PATH, 'r') as fl:
    L = [[int(x) for x in row.split(',')] for row in fl]

    
A = matrix(L).T
L = IntegerLattice(A)

v = '''
[-123691   61793  -54109   33688   55390   31796  184384  177272
  169211  157512  235425  336389   43560 -186904   98831   -8870
  143692 -181223 -294567  -53475  -64268   48353  -66970  173423
  150737  127174   40694  174755   85646 -273670   68170  101793
 -197802  292172  160773   73080 -126291  238470 -180629  236333
  153284  245015 -233516  -64198 -184434 -238536  294854  283913
  -81797  -36811 -274979  297269  -48197   46353 -104330  -93276
  143649  290728   70673  -99554  -12315 -188475  -54067  139848
 -188974 -173744   69786   25712  -72084   86034]
'''
v = vector(tuple(int(x) for x in v[2:-2].split()))
print(v)
print(v in L)
