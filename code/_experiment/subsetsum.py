import numpy as np
from sklearn.neural_network import MLPClassifier

RND = np.random.default_rng(seed=3)

set_cardinality = 30
examples = 2**10
upper_bound = 2**8
testing_ratio = 0.33

M = RND.integers(upper_bound, size=set_cardinality, dtype=np.ulonglong)
X = RND.integers(2, size=(set_cardinality, examples), dtype=M.dtype)
X[0][::2], X[0][1::2] = 0, 1
S = M @ X

data = np.hstack((np.repeat([M], S.shape[0], axis=0), S.reshape(-1, 1)))
labels = X[0]

threshold = labels.shape[0] - int(testing_ratio * labels.shape[0])
train_X = data[:threshold]
train_y = labels[:threshold]
test_X = data[threshold:]
test_y = labels[threshold:]

best = 0
for n in range(1, 101):
    network = MLPClassifier(random_state=0,
                            verbose=False,
                            max_iter=2**32,
                            hidden_layer_sizes=n)
    network.fit(train_X, train_y)
    pred_y = network.predict(test_X)
    cur = np.sum(test_y == pred_y) / test_y.shape[0]
    if (cur > best):
        best = cur
        print(f'{n:<3d} hidden units with accuracy {best:3f}')
    else:
        print(f'{n:<3d}', end='\r')
