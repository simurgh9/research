import numpy as np
from net.ffnn import FFNN
from net.ffnn_parallel import ParallelBatch, ParallelExample  # noqa: F401
from time import time as seconds


def generic_parallel_time_tests(epochs, ParallelNetwork):
    """Generic test format to compare running times of networks with
    parallel implementation in Python vs. serial.
    """

    def f(X):  # polynomial to learn
        return X**4 - 22 * X**2

    X, X_test = np.linspace(-5, 5, 10000), np.linspace(-5, 5, 500)
    y, _ = f(X), f(X_test)
    net_serial = FFNN((X, y), [1, 100, 1], epk=epochs, eta=0.0001, is_reg=True)
    net_parallel = ParallelNetwork((X, y), [1, 100, 1],
                                   epk=epochs,
                                   eta=0.0001,
                                   is_reg=True)

    start = seconds()
    train_mse_serial = net_serial.tango()
    duration_serial = seconds() - start

    start = seconds()
    train_mse_parallel = net_parallel.tango()
    duration_parallel = seconds() - start

    return (duration_serial, train_mse_serial, duration_parallel,
            train_mse_parallel)


def test_parallel_example():
    """ParallelExample is overall slower than serial due to Process
    class start-up overhead. And other Python GIL things. Just Google.

    """
    results = generic_parallel_time_tests(5, ParallelExample)
    duration_serial, train_mse_serial = results[0], results[1]
    duration_parallel, train_mse_parallel = results[2], results[3]
    assert duration_parallel > duration_serial
    assert abs(train_mse_serial - train_mse_parallel) < 0.00001


def test_parallel_batch():
    """ParallelBatch does not always run faster, e. g., when training to
    classify handwritten digits. It actually runs slower than FFNN.

    ParallelBatch does not converge nicely because weights do not
    update after each batch, instead after each epoch. That is why
    train_mse_serial is better (i. e., smaller than)
    train_mse_parallel.
    """
    results = generic_parallel_time_tests(10, ParallelBatch)
    duration_serial, train_mse_serial = results[0], results[1]
    duration_parallel, train_mse_parallel = results[2], results[3]
    assert abs(duration_parallel - duration_serial) < 5  # seconds
    assert train_mse_serial < train_mse_parallel
