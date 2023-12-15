import numpy as np
from urllib.request import urlopen as get

dim = 100  # https://latticechallenge.org/svp-challenge/
url = f'https://latticechallenge.org/svp-challenge/download/challenges/svpchallengedim{dim}seed0.txt'
with get(url) as response:
    text = response.read().decode('utf-8')
    rows = text.replace('[', '').replace(']', '').split('\n')[:-2]
    L = [[int(x) for x in row.split(' ')] for row in rows]

np.savetxt(f'challenge.csv', L, fmt='%d', delimiter=',')
