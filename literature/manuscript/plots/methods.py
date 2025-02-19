import numpy as np

data = np.genfromtxt(
    'algo_stats.tsv',
    delimiter='\t',
    dtype=str)
data = data[:, [1, 2, 3, 6]]
count = 0
total = 0
for method in data[:, -1][1:]:
    total+= 1
    if 'sieving' in method.lower():
        count += 1

print(f'{count} / {total}')
    
