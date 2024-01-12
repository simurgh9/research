import numpy as np
from net.ffnn import FFNN
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt

plt.tight_layout()


def f(X):  # function to learn
    return X**4 - 22 * X**2
    # return np.sin(X)


c = 1.8
lo, hi = -c * np.pi, c * np.pi
X, X_test = np.linspace(lo, hi, 1000), np.linspace(lo, hi, 500)
y, y_test = f(X), f(X_test)


def report():
    preds = net.predict(X, lambda x: x[0])
    plt.plot(X, y, 'ro', label='Expected')
    plt.plot(X, preds, 'c.', label='Predictions')
    plt.legend()
    plt.draw()
    plt.pause(1e-7)
    plt.cla()


net = FFNN((X, y), [1, 100, 1], 19, is_reg=True)
net_sk = MLPRegressor(hidden_layer_sizes=(100, ),
                      activation='logistic',
                      solver='sgd',
                      alpha=0.0001,
                      batch_size=32,
                      learning_rate='constant',
                      learning_rate_init=0.001,
                      max_iter=1000,
                      shuffle=True,
                      random_state=0,
                      verbose=False,
                      momentum=0.9,
                      nesterovs_momentum=False,
                      validation_fraction=0)
net_sk.fit(X.reshape(-1, 1), y)
net.tango(report)

preds = net.predict(X_test, lambda x: x[0])
preds_sk = net_sk.predict(X_test.reshape(-1, 1))
plt.plot(X_test, y_test, 'ro', label='Expected')
plt.plot(X_test, preds, 'c.', label='Predictions')
plt.plot(X_test, preds_sk, 'k.', label='Predictions SKlearn')
plt.legend()
plt.show()
