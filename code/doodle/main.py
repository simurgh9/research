import numpy as np

M, S = np.array([2, 3, 4, 9, 14, 23]), 21
n = len(M)
x = np.array([0, 1, 1, 0, 1, 0])
L = 2*np.identity(n)  # matrix with 2's along the diagonal
L = np.vstack((L, np.ones(n)))  # adding ones to the bottom
L = np.hstack((L, np.hstack((M, S)).reshape(-1, 1))).astype(np.int64)
a = np.hstack((x, -1))
t = (a @ L).astype(np.int64)
print(f'M = {M}, S = {S}')
print(f'The lattice matrix L:\n{L}')
print(f'The linear commbination vector a = {a}.')
print(f't = aL = {t}')
