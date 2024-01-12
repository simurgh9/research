import numpy as np
from urllib.request import urlopen as get

# https://latticechallenge.org/svp-challenge/
for dim in range(40, 200, 2):
    url = f'https://latticechallenge.org/svp-challenge/download/challenges/svpchallengedim{dim}seed0.txt'
    with get(url) as response:
        text = response.read().decode('utf-8')
        rows = text.replace('[', '').replace(']', '').split('\n')[:-2]
        L = [[int(x) for x in row.split(' ')] for row in rows]
    np.savetxt(f'../../../saved/bases/challenge{dim}.csv', L, fmt='%d', delimiter=',')
