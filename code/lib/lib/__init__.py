import numpy as np
from matplotlib import pyplot as plt
from math import floor, log2
from pkgutil import get_data

RND = np.random.default_rng(seed=0)
AVAILABLE_MEMORY = 24
minimum = 'Least accuratly factored number was with: {:.2%} correct bits.'
mean = 'On average a number factored with: {:.2%} correct bits.'
maximum = 'Most accuratly factored number was with: {:.2%} correct bits.'


def primes32bit():
    half_gaps = get_data(__package__, 'data/half_gaps.bin')
    half_gaps = np.frombuffer(half_gaps, dtype=np.uint8)  # raw bytes (8-bits)
    primes = 2 * np.cumsum(half_gaps, dtype=np.uint32) + 3
    return np.hstack(([2, 3], primes)).astype(np.uint32, copy=False)


def semiprime(arr):
    arr = arr if (arr.shape[0] % 2) == 0 else arr[:-1]
    try:
        p, q = np.split(arr.astype(np.ulonglong, copy=False), 2)
    except OverflowError:
        p, q = np.split(arr, 2)
    return p * q, p


def compare_complementes(y, preds):
    complement = (preds + 1) % 2
    corrects = (y == preds).sum(axis=1)
    corrects_comp = (y == complement).sum(axis=1)
    # https://stackoverflow.com/a/72609922/12035739
    return np.where((corrects >= corrects_comp)[:, None], preds, complement)


def prep_as_data(N,
                 p,
                 train_ratio=0.6,
                 shuffle=True,
                 classification=True,
                 sklearn=False):
    if shuffle:
        shuffle_ind = RND.permutation(N.shape[0])
        N, p = N[shuffle_ind], p[shuffle_ind]
    if classification:
        bits = floor(log2(N.max())) + 1
        gb = (8 * bits * N.shape[0]) / (2**30)  # 2**30 is one gibibyte
        if gb >= AVAILABLE_MEMORY:
            raise MemoryError(f'Data takes {gb:.1f} Gibibytes.')
        N, p = to_bin(N), to_bin(p)
    if not classification and sklearn:
        N = N.reshape(-1, 1)
    at = int(train_ratio * N.shape[0])
    X, y = N[:at], p[:at]  # train
    _X, _y = N[at:], p[at:]  # test
    return X, y, _X, _y


def to_bin(ls):
    bits_needed = 1
    if max(ls) > 0:
        bits_needed = int(max(ls)).bit_length()
    dtype = np.ulonglong if bits_needed < 65 else object
    arr = np.array(ls, dtype=dtype)
    powers = 2**np.arange(bits_needed - 1, -1, -1, dtype=dtype)
    columnised = arr.reshape(-1, 1)
    bitwise_and = columnised & powers  # [2] & [4, 2, 1] = [0, 2, 0]
    binary = (bitwise_and > 0).astype(np.uint8)
    return binary if binary.shape[0] > 1 else binary[0]


def to_dec(ls):
    ls = np.array(ls)
    bits_needed = ls.shape[0]
    if len(ls.shape) > 1:  # is a binary matrix
        bits_needed = ls.shape[1]
    dtype = np.ulonglong if bits_needed < 65 else object
    arr = ls.astype(dtype, copy=False)
    powers = 2**np.arange(bits_needed - 1, -1, -1, dtype=dtype)
    powers = powers.reshape(-1, 1)
    decimal = (arr @ powers).flatten()
    return decimal if decimal.shape[0] > 1 else decimal[0]


def print_semiprime_stats(N, p):
    print('Log information for N:')
    try:
        n_bits = np.log2(N)
    except TypeError:
        n_bits = np.array([log2(n) for n in N])
    print(f'\tlg(max(N))  {chr(8776)} {n_bits.max():.2f}')
    print(f'\tlg(mean(N)) {chr(8776)} {n_bits.mean():.2f}')
    print(f'\tlg(min(N))  {chr(8776)} {n_bits.min():.2f}')
    print('Log information for p:')
    try:
        p_bits = np.log2(p)
    except TypeError:
        p_bits = np.array([log2(factor) for factor in p])
    print(f'\tlg(max(p))  {chr(8776)} {p_bits.max():.2f}')
    print(f'\tlg(mean(p)) {chr(8776)} {p_bits.mean():.2f}')
    print(f'\tlg(min(p))  {chr(8776)} {p_bits.min():.2f}')
    return n_bits, p_bits


def bitwise_accuracy_graph(y, preds):
    bitwise_acc = ((y == preds).sum(axis=0) / y.shape[0])
    ones = (y.sum(axis=0) / y.shape[0])
    zeros = 1 - ones
    idx = np.arange(y.shape[1])
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10))  # inches
    fig.tight_layout(pad=4)
    if y.shape[1] > 100:
        ax2.plot(zeros[::-1], 'ro', fillstyle='none', alpha=0.5, label='Zeros')
        ax2.plot(ones[::-1], 'gx', alpha=0.5, label='Ones')
        ax2.plot(bitwise_acc[::-1], 'c.', label='Accuracy')
    else:
        ax2.bar(idx, bitwise_acc[::-1], color='c', width=0.25, label='Acc.')
        ax2.bar(idx - 0.25, zeros[::-1], color='r', width=0.25, label='Zeros')
        ax2.bar(idx + 0.25, ones[::-1], color='g', width=0.25, label='Ones')
    ax2.set(xlabel='$i^{th}$ bit (least sig. on right)',
            ylabel='Ratio/Mean Accuracy',
            xlim=(idx[-1] + 0.5, -0.5),
            title=('Mean accuracy for the $i^{{th}}$ bit of ' +
                   f'{y.shape[1]}-bit test-examples $y = p$.'))
    ax2.legend()
    return fig, ax1


def examplewise_accuracy_graph(y, preds, fig, ax):
    pred_rand = RND.choice([0, 1], size=y.shape).astype(np.uint8)
    # should i compare complementes for the random baseline?
    # pred_rand = compare_complementes(y, pred_rand)
    correct_bits_per_example = (y == preds).sum(axis=1) / y.shape[1]
    correct_bits_per_example_random = (y == pred_rand).sum(axis=1) / y.shape[1]
    correct_bits_per_example.sort()
    correct_bits_per_example_random.sort()
    ax.plot(correct_bits_per_example, 'k.', label='Network')
    ax.plot(correct_bits_per_example_random, 'r.', label='Random')
    total_bits, total_ones = np.prod(y.shape), y.sum()
    ax.set(
        xlabel='Index $j$ for each test-example $y_j = p_j$ in test set.',
        ylabel='Acc. for the $j^{{th}}$ test-example $y_j = p_j$.',
        title=('Individual accuracies of ' +
               f'{y.shape[0]} test-examples $y_j = p_j$.\n' +
               f'Min: {correct_bits_per_example.min():.2%}, ' +
               f'Mean: {correct_bits_per_example.mean():.2%}, ' +
               f'Max: {correct_bits_per_example.max():.2%} ' +
               f'Mean (random): {correct_bits_per_example_random.mean():.2%}, '
               f'{total_ones/total_bits:.2%} of all test-example bits are 1.'))
    ax.legend()
    fig.savefig('../../saved/figures/last_plot.png')
    plt.show()


def xgcd(a, b):
    (a, b) = (b, a) if a < b else (a, b)
    v1, v2 = np.array([1, 0]), np.array([0, 1])
    while b > 0:
        quotient, remainder = divmod(a, b)
        a, b = b, remainder
        v1, v2 = v2, v1 - (quotient * v2)
    return a, v1[0], v1[1]


def inv_mod(x, m):
    x_ = x % m
    g, _, b = xgcd(m, x_)
    if g != 1:
        raise AttributeError(f'gcd({x}, {m}) is not 1.')
    return b % m
