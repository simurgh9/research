import numpy as np
from sys import argv
from time import perf_counter
from matplotlib import pyplot as plt
from lib import primes32bit, prep_as_data
from sklearn.neural_network import MLPClassifier

RNG = np.random.default_rng(0)
primes = primes32bit()


def n_bit_data(b):
    if b < 4:  # there are no odd composite numbers less than 8
        raise ValueError("Specify bits greater than 4.")
    begin = perf_counter()
    naturals = 2 * np.arange(1, 2**(b - 1)) + 1  # odd x in N of b bits
    naturals = naturals[naturals % 5 != 0]  # deleting x if x is 0 mod 5
    map_primes = np.isin(naturals, primes, assume_unique=False)
    idx_composites = np.where(map_primes == 1)[0] + 1  # composites near primes
    idx_composites = idx_composites[
        idx_composites < naturals.shape[0]]  # if last natural is prime
    trimmed_p = naturals[map_primes]
    composites = naturals[idx_composites]  # may still have twin primes
    not_twin = np.isin(composites, trimmed_p, invert=True, assume_unique=False)
    composites = composites[not_twin]  # deleting twin primes
    trimmed_p = trimmed_p[np.where(not_twin == 1)]
    X = np.hstack((trimmed_p, composites))
    is_prime = np.ones(shape=composites.shape)
    is_not_prime = np.zeros(shape=composites.shape)
    y = np.hstack((is_prime, is_not_prime))
    sorted_map = np.argsort(X)
    # print(f'Data generation took: {np.ceil(perf_counter() - begin):2.0f}s.')
    return X[sorted_map], y[sorted_map].astype(np.uint8)


def ffnn(hidden):  # https://stackoverflow.com/a/46913369/12035739
    return MLPClassifier(momentum=0.1,
                         shuffle=False,
                         max_iter=2**10,
                         activation='tanh',
                         random_state=hidden,
                         early_stopping=True,
                         learning_rate_init=2 / b,
                         hidden_layer_sizes=hidden)


low, high, epsilon = 7, int(argv[1]), 0.56
bits, hidden = list(range(low, high + 1)), []
best_acc = []
for b in bits:
    X, y = n_bit_data(b)
    X, _, _, _ = prep_as_data(X, y, 1, shuffle=False)
    X = X.astype(np.ulonglong)
    y = np.eye(2)[y.reshape(-1)].astype(np.ulonglong)
    neurons, scr, max_scr = 0, 0, (0, 0)
    while scr < epsilon:
        neurons += 1
        if neurons > 100:
            neurons = -1
            break
        clf = ffnn(neurons).fit(X, y)
        # scr = clf.validation_scores_[-1]
        scr = (y[:, 1] == np.argmax(clf.predict(X), axis=1)).mean()  # accuracy
        max_scr = (neurons, scr) if max_scr[1] < scr else max_scr
        # print(f'{b:3} {neurons:3} {scr:3.4f} {max_scr}', end='\r')
    hidden += [neurons]
    best_acc += [max_scr]
    print(f'{b:3} {neurons:3} {scr:3.4f} {max_scr}')

bits, hidden = np.array(bits), np.array(hidden)
plt.xlabel('Maximum Number of Bits')
plt.ylabel('Number of Neurons in the Hidden Layer')
plt.plot(bits[hidden > 0], hidden[hidden > 0])
plt.savefig('../../saved/figures/last_plot_primality.png')
# plt.show()
plt.clf()
plt.xlabel('Number of Neurons in the Hidden Layer')
plt.ylabel('Accuracy (Baseline 50%)')
plt.xticks(bits, [f'{b},{best_acc[i][0]}' for i, b in enumerate(bits)], rotation=70)
plt.plot(bits, [e[1] for e in best_acc])
plt.savefig('../../saved/figures/last_plot_primality_acc.png')
# plt.show()
