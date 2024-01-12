import numpy as np
from sklearn.neural_network import MLPClassifier

RND = np.random.default_rng(seed=0)

set_cardinality = 100
examples = 2**10
upper_bound = 2**8
testing_ratio = 0.33
index_to_question = 0  # whether the 0th element of M is in the sum
M = RND.integers(upper_bound, size=set_cardinality, dtype=np.ulonglong)
X = RND.integers(2, size=(set_cardinality, examples), dtype=M.dtype)
X[index_to_question][::2], X[index_to_question][1::2] = 0, 1
S = M @ X

data = np.hstack((np.repeat([M], S.shape[0], axis=0), S.reshape(-1, 1)))
labels = X[0]

at = labels.shape[0] - int(testing_ratio * labels.shape[0])
X, y, _X, _y = data[:at], labels[:at], data[at:], labels[at:]

best, accumulate = 0, 0
for n in range(1, 5 * set_cardinality):
    network = MLPClassifier(n, random_state=0, verbose=False, max_iter=2**32)
    network.fit(X, y)
    cur = np.sum(network.predict(_X) == _y) / _y.shape[0]
    accumulate += cur
    if (cur > best):
        best = cur
        print(f'{n:<3d} hidden units, accuracy: {best:3f}')
    else:
        print(f'{n:<3d} hidden units, average: {accumulate / n:3f}', end='\r')
