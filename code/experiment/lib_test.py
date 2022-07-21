import lib as lb
import numpy as np


def test_primes32bit():
    primes = lb.primes32bit()
    maximum_gap = (primes[1:] - primes[:-1]).max()
    assert maximum_gap == 336
    assert primes[0] == 2
    assert primes[-1] == 4294967291
    assert primes.shape == (203280221, )
    assert primes.dtype == np.uint32


def test_semiprime():
    primes = lb.primes32bit()
    primes = primes if (primes.shape[0] % 2) == 0 else primes[:-1]
    N, _p = lb.semiprime(primes)
    halfway = primes.shape[0] // 2
    p = primes[:halfway].astype(np.ulonglong)
    q = primes[halfway:].astype(np.ulonglong)
    assert (p == _p).all()
    assert (N == p * q).all()
    assert (N % _p).sum() == 0


def test_compare_complements():
    y = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 1],
                  [0, 0, 1, 0]])
    preds = np.array([[0, 0, 0, 0],
                      [1, 1, 1, 0],
                      [1, 1, 0, 0],
                      [1, 1, 1, 1],
                      [0, 0, 1, 0]])
    expectation = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 1],
                            [1, 1, 1, 1],
                            [0, 0, 1, 0]])
    assert (lb.compare_complementes(y, preds) == expectation).all()


def test_prep_as_date():
    primes = lb.primes32bit()[:200]
    N, p = lb.semiprime(primes)
    X, y, _X, _y = lb.prep_as_data(N, p)
    assert X.shape[0] == 60 and y.shape[0] == 60
    assert _X.shape[0] == 40 and _y.shape[0] == 40
    X, y, _X, _y = lb.prep_as_data(N, p, train_ratio=0)
    assert X.shape[0] == 0 and y.shape[0] == 0
    assert _X.shape[0] == 100 and _y.shape[0] == 100
    X, y, _X, _y = lb.prep_as_data(N, p, train_ratio=1)
    assert X.shape[0] == 100 and y.shape[0] == 100
    assert _X.shape[0] == 0 and _y.shape[0] == 0


def test_to_bin():
    assert lb.to_bin([0, 0, 0]).sum() == 0
    for i, bits in enumerate(lb.to_bin(range(8))):
        string = ''.join([str(b) for b in bits])
        assert int(string, 2) == i
    all_ones = lb.to_bin([(2**65) - 1])
    single_one = lb.to_bin([2**65])
    assert all_ones.sum() == 65
    assert single_one.sum() == 1
    assert single_one[0] == 1
    assert all_ones.dtype == single_one.dtype == np.uint8


def test_to_dec():
    ls = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    for i, n in enumerate(lb.to_dec(ls)):
        assert i == n
    bits64 = [1 for i in range(64)]  # 2**64 - 1
    bits64 = lb.to_dec(bits64)
    bits65 = [1 for i in range(65)]  # 2**65 - 1
    bits65 = lb.to_dec(bits65)
    assert bits64 == (2**64 - 1) and type(bits64) == np.ulonglong
    assert bits65 == (2**65 - 1) and type(bits65) == int


def test_print_semiprime_stats():
    primes = lb.primes32bit()[-200:]
    primes = primes if (primes.shape[0] % 2) == 0 else primes[:-1]
    N, p = lb.semiprime(primes)
    n_bits, p_bits = lb.print_semiprime_stats(N, p)
    halfway = primes.shape[0] // 2
    p, q = primes[:halfway], primes[halfway:]
    for factor, bits in zip(p, p_bits):
        assert np.log2(factor) == bits
    for fp, fq, bits in zip(p, q, n_bits):
        assert abs(bits - np.log2(fp) - np.log2(fq)) < 0.0001
