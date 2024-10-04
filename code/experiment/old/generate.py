import numpy as np
from math import floor
from lib import primes32bit
from decimal import Decimal, getcontext 

getcontext().prec = 103

n = 100
ID = 113083156
S = 2 * ID * (10**94)
a = 10**100
b = 1 / Decimal(3)
M = [floor(a * (p**b)) for p in primes32bit()[:n]]

k = 2*np.identity(n, dtype=int)
k = np.column_stack((k, np.ones(n, dtype=int)))
k = np.row_stack((k, M + [S]))
# print(k)

for i in range(n+1):
    for j in range(n):
        print(k[i][j], end=',')
    print(k[i][-1])
