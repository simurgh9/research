from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from lib import semiprime, prep_as_data, primes32bit
from pickle import dump


GRID_SEARCH_SUBSET = 2**12
primes_numbers = primes32bit()
N, p = semiprime(primes_numbers)
X, y, _X, _y = prep_as_data(N, p, train_ratio=0.625)

# smaller chunk for grid searching hyper-parameters
grid_N, grid_p = semiprime(primes_numbers[:GRID_SEARCH_SUBSET])
grid_X, grid_y, _, _ = prep_as_data(grid_N, grid_p, train_ratio=1)
param_sp = {
    'hidden_layer_sizes': [100, 64],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5],
    'learning_rate_init': [0.001, 0.01, 0.1, 0.5],
    'activation': ['relu', 'logistic', 'tanh']
}


def scorer(y, _y):
    return (y == _y).sum() / (y.shape[0] * y.shape[1])


custom_scorer = make_scorer(scorer, greater_is_better=True)
network = MLPClassifier(random_state=0, max_iter=2**32, solver='adam')
search = GridSearchCV(network,
                      param_sp,
                      n_jobs=-1,
                      cv=3,
                      scoring=custom_scorer,
                      verbose=1)
search.fit(grid_X, grid_y)
print(f'Best parameters found with score: {search.best_score_}')
print(search.best_params_)

network = search.best_estimator_
network.verbose = True
network.fit(X, y)
print(f'ANN retrained with test score: {scorer(_y, network.predict(_X))}')

with open('../../../saved/model_con_new.p', 'wb') as fl:
    dump(network, fl)
