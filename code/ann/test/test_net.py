import numpy as np
from net.ffnn import FFNN
from net.mnsit import MNSIT
from matplotlib import pyplot as plt
from test_mnsit import PATH

plt.tight_layout()

def test_polynomial_regression():
    """Tests the network's ability to approximate a polynomial."""

    def f(X):  # polynomial to learn
        return X**4 - 22 * X**2

    X, X_test = np.linspace(-5, 5, 10000), np.linspace(-5, 5, 500)
    y, y_test = f(X), f(X_test)
    net = FFNN((X, y), [1, 100, 1], epk=5, eta=0.002, is_reg=True)

    def report():
        preds = net.predict(X, lambda x: x[0])
        plt.plot(X, y, 'g.')
        plt.plot(X, preds, 'r.')
        plt.draw()
        plt.pause(1e-7)
        plt.cla()

    # train_mse = net.tango(report)
    train_mse = net.tango()
    preds = net.predict(X_test, lambda x: x[0])
    test_mse = net.squared_error(y_test, preds, derivative=False).mean()
    # plt.title(f'MSE = {test_mse}.')
    # plt.plot(X_test, y_test, 'g.', label='Expected')
    # plt.plot(X_test, preds, 'r.', label='Predictions')
    # plt.legend(), plt.show()
    assert train_mse < 13 and test_mse < 8


def test_trigonometric_regression():
    """Tests the network's ability to approximate a trigonometric function."""

    X = np.linspace(-np.pi, np.pi, 1000)
    X_test = np.linspace(-np.pi, np.pi, 500)
    y, y_test = np.sin(X), np.sin(X_test)
    net = FFNN((X, y), [1, 100, 1], epk=57, is_reg=True)

    train_mse = net.tango()
    preds = net.predict(X_test, lambda x: x[0])
    test_mse = net.squared_error(y_test, preds, derivative=False).mean()
    assert train_mse < 0.02 and test_mse < 0.02


def test_parity_classification():
    """Tests the network's ability to classify numbers even or odd."""

    X = np.array([[int(b) for b in f'{e:010b}'] for e in range(2**10)])
    y = np.array([int(e % 2 == 0) for e in range(2**10)])
    X, y, X_test, y_test = X[:700], y[:700], X[700:], y[700:]

    net = FFNN((X, y), [10, 3, 2], epk=6, bt=10, eta=0.5)
    net.one_hot_labels()
    train_mse = net.tango()

    preds = net.predict(X_test, f=np.argmax)
    acc = np.sum(preds == y_test) / len(y_test)
    assert train_mse < 0.0000005 and acc > 0.98


def test_mnsit_classification_pretrained():
    """Tests the network's ability to classify handwritten digits
    using pre-trained weights and biases.
    """

    mn = MNSIT(path=PATH+'mnsit_data/')
    X, X_test, y, y_test = mn.get_data()
    # only one epoch because the weights and biases are already trained.
    net = FFNN((X, y), [784, 32, 10], epk=1, bt=256, eta=0.3, alpha=0)
    net.one_hot_labels()
    net.load_weights_biases(PATH+'weights_biases.npy')
    train_mse = net.tango()
    preds = net.predict(X_test, f=np.argmax)
    acc = np.sum(preds == y_test) / len(y_test)
    print(train_mse, acc)
    assert train_mse < 0.02 and acc > 0.9


def test_mnsit_classification():
    """Tests the network's ability to classify handwritten digits."""

    mn = MNSIT(path=PATH+'mnsit_data/')
    X, X_test, y, y_test = mn.get_data()
    net = FFNN((X, y), [784, 32, 10], epk=6, bt=256, eta=0.3, l2=0, alpha=0)
    net.one_hot_labels()
    train_mse = net.tango()
    preds = net.predict(X_test, f=np.argmax)
    acc = np.sum(preds == y_test) / len(y_test)
    assert train_mse < 0.065 and acc > 0.50  # baselines is 1/10
