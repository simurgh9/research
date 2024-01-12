import numpy as np
from time import time as seconds


class FFNN:

    def __init__(self, data, topo, epk, bt=32, eta=0.001,
                 alpha=0.9, l2=0.0001, is_reg=False):
        """Feed-forward Neural Network.

        Args:
            data (tuple): Training examples and labels of form (X, y).
            topo (list): Architecture of the network.
            epk (int): Number of epochs to run for training.
            bt (int): Batch size for Stochastic Gradient Descent.
            eta (float): Learning rate.
            alpha (float): Momentum.
            sigma (float): Standard deviation for intial weights & biases.
            mu (float): Mean for intial weights & biases.
            l2 (float): Strength of the L2 regularization term.
            is_reg (bool): If we are performing regression.
        """
        self.RND = np.random.default_rng(seed=0)
        self.X, self.y = data
        self.topo = topo
        self.epk, self.bt, self.eta, self.alpha = epk, bt, eta, alpha
        self.l2 = l2
        self.is_reg = is_reg
        self.Wb = self.random_weights_biases()
        self.momentum = 0 * self.Wb

    def random_weights_biases(self, sigma=1, mu=0):
        Wb = np.empty(2, dtype=np.ndarray)
        Wb[0] = np.empty(self.L() - 1, dtype=np.ndarray)
        Wb[1] = np.empty(self.L() - 1, dtype=np.ndarray)
        for i in range(self.L() - 1):
            c, r = self.topo[i], self.topo[i + 1]
            bound = np.sqrt(2.0 / (c + r))
            Wb[0][i] = self.RND.uniform(-bound, bound, (r, c))
            Wb[1][i] = self.RND.uniform(-bound, bound, r)
        return Wb

    def tango(self, report=lambda: None, print_steps=False):
        for epoch in range(self.epk):
            start, error = seconds(), self.SGD(print_steps)
            duration = seconds() - start
            print(f'{epoch:4d}. MSE: {error:<10.5g} Time: {duration:3.5g}sec.')
            report()
        return error  # error of the last epoch

    def SGD(self, print_steps):  # Stochastic Gradient Descent
        error_sum_over_bt = 0
        for i, mini_batch in enumerate(self.batches()):
            start = seconds()
            grad_bt, error_bt = self.gradient_bt(mini_batch)
            duration = seconds() - start
            error_sum_over_bt += error_bt
            regulariser = self.l2 * self.Wb  # ASK IF YOU'RE DOING THIS RIGHT
            self.Wb = self.Wb - self.learning_rate() * grad_bt + regulariser
            if i > 0:
                self.Wb = self.Wb - self.mom_rate() * (self.momentum / i)
            self.momentum += self.mom_rate() * (self.learning_rate() * grad_bt)
            if print_steps:
                print(f'SGD error: {error_bt:3.5g} Time: {duration:3.5g}s')
        return error_sum_over_bt / self.total_bt()  # mean of all batch errors

    def gradient_bt(self, mini_batch):
        nabla, error_sum_over_x = self.backpropagation(*next(mini_batch))
        for x, y in mini_batch:
            gradient, error = self.backpropagation(x, y)
            error_sum_over_x += error
            nabla += gradient
        # Average of:
        #     - gradients
        #     - mean (over activations) squared error:
        #         + mean(sum(square(y - activations)))
        # over x in mini_batch
        return nabla / self.bt, error_sum_over_x / self.bt

    def backpropagation(self, x, y):
        outputs, activations = self.forward_pass(x)
        gradient = self.backward_pass(outputs, activations, y)
        return gradient, self.mean_example_error(y, activations[-1])

    def forward_pass(self, example, keep_track=True):
        curr_layer = example.flatten()
        outputs = np.empty(self.L() - 1, dtype=np.ndarray)  # z^(l)
        activations = np.empty(self.L(), dtype=np.ndarray)  # a^(l)
        activations[0] = curr_layer
        for W, b, l in zip(self.W(), self.b(), range(self.L() - 1)):
            outputs[l] = (W @ curr_layer) + b
            if self.is_reg and (l == self.L() - 2):  # last layer regression
                activations[l + 1] = outputs[l]
                continue  # we should be done
            activations[l + 1] = self.act(outputs[l])
            curr_layer = activations[l + 1]
        return (outputs, activations) if keep_track else activations[-1]

    def backward_pass(self, outputs, activations, y):
        gradient_W = np.empty(self.L() - 1, dtype=np.ndarray)
        gradient_b = np.empty(self.L() - 1, dtype=np.ndarray)
        z, a = outputs[-1], activations[-1]  # z^L, a^L
        delta = -1 * self.error(y, a, derivative=True)
        if not self.is_reg:
            delta *= self.act(z, derivative=True)  # delta^L eq 2
        delta = delta.reshape((1, delta.shape[0]))
        for l in range(self.L() - 2, -1, -1):  # noqa: E741
            a_prev = activations[l]
            a_prev = a_prev.reshape((len(a_prev), 1)).T
            pC_w, pC_b = (delta.T @ a_prev), delta.flatten()
            gradient_W[l], gradient_b[l] = pC_w, pC_b
            if l > 0:  # we have no layers behind l = 0
                z, a = outputs[l - 1], activations[l]
                delta = (delta @ self.W()[l]) * self.act(z, True)
        gradient = np.empty(shape=2, dtype=np.ndarray)
        gradient[0], gradient[1] = gradient_W, gradient_b
        return gradient

    def act(self, x, derivative=False):
        return self.sigmoid(x, derivative)

    def sigmoid(self, x, derivative=False):
        x[x < -10], x[x > 10] = -10, 10  # avoiding overflows in np.exp
        s = lambda x: 1 / (1 + np.exp(-x))  # noqa E731
        return s(x) * (1 - s(x)) if derivative else s(x)

    def batches(self):
        shuffle_ind = self.RND.permutation(self.X.shape[0])
        shuffle_X, shuffle_y = self.X[shuffle_ind], self.y[shuffle_ind]
        for i in range(self.total_bt()):  # index out of bound = last index
            l, u = i * self.bt, (i + 1) * self.bt
            mini_X, mini_y = shuffle_X[l:u], shuffle_y[l:u]
            yield zip(mini_X, mini_y)

    def total_bt(self):
        return int(np.ceil(self.X.shape[0] / self.bt))

    def learning_rate(self):
        return self.eta

    def mom_rate(self):
        return self.alpha

    def L(self):
        return len(self.topo)

    def W(self):
        return self.Wb[0]

    def b(self):
        return self.Wb[1]

    def one_hot_labels(self):
        self.y = np.eye(len(set(self.y)))[self.y.reshape(-1)]

    def mean_example_error(self, y, a):
        er = np.mean(self.error(y, a))
        if er > 1e8:
            notice = f'Example error: {er:g} too large, try eta < {self.eta}.'
            raise ValueError(notice)
        return er

    def error(self, y, a, derivative=False):
        return self.squared_error(y, a, derivative)

    def squared_error(self, y, a, derivative):
        return 2 * (y - a) if derivative else np.square(y - a)

    def predict(self, examples, f=lambda x: x):
        try:
            return np.array([f(self.forward_pass(x, False)) for x in examples])
        except TypeError:
            return f(self.forward_pass(examples, False))

    def __repr__(self):
        ret = ''
        for l, W, b in zip(self.topo, self.W(), self.b()):  # noqa: E741
            ret += f'W{W.shape}x({l}, 1) + b({b.shape[0]}, 1)\n'
        return ret

    def __str__(self):
        return self.__repr__()

    def save_weights_biases(self, path='./weights_biases.npy'):
        return np.save(path, self.Wb)

    def load_weights_biases(self, path='./weights_biases.npy'):
        self.Wb = np.load(path, allow_pickle=True)
