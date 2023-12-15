import numpy as np
from lib.lattice import sieve, population

# B = np.array([[3, 0], [0, 7]])
# B = np.array([[95, 47], [460, 215]])
B = np.loadtxt('columns.csv', delimiter=',')
dims = {2: 4, 40: 650, 50: 700, 60:9000, 70:100000, 100:100000}
size =  dims[100]

P = population(B, size)
P = np.array(sieve(P, size), dtype=np.int64).T
idx = np.argsort(np.sum(P**2, axis=0))
print(P[:, idx][:, 0])
