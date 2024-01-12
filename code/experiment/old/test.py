import numpy as np
from lib import bitwise_accuracy_graph, examplewise_accuracy_graph
from lib import minimum, mean, maximum, primes32bit, compare_complementes
from lib import semiprime, prep_as_data, print_semiprime_stats, to_dec

primes_numbers = primes32bit()[:2**25]
network = np.load('../../../saved/model_con.p', allow_pickle=True)

# primes_numbers = primes32bit()
# network = np.load('../../saved/model_con_full.p', allow_pickle=True)

# primes_numbers = np.load('../../saved/erratic.npy', allow_pickle=True)
# network = np.load('../../saved/model_err.p', allow_pickle=True)

N, p = semiprime(primes_numbers)
X, y, _X, _y = prep_as_data(N, p, train_ratio=0.625)
print_semiprime_stats(N, p)

# predictions by the network
preds = np.ceil(network.predict_proba(_X) - 0.5).astype(np.uint8)
preds = compare_complementes(_y, preds)
correct_bits_per_example = (_y == preds).sum(axis=1) / _y.shape[1]

# printing testing results results
print(minimum.format(correct_bits_per_example.min()))
print(mean.format(correct_bits_per_example.mean()))
print(maximum.format(correct_bits_per_example.max()))

# plotting testing results
fig, ax = bitwise_accuracy_graph(_y, preds)
examplewise_accuracy_graph(_y, preds, fig, ax)

# using the most accuratly predicted p to factor its corresponding N
idx = correct_bits_per_example.argmax()
N, p = to_dec(_X[idx]), to_dec(preds[idx])
q, r = divmod(N, p)
print(f'p={p}\nq={q}\nremainder={r}')
