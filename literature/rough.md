## Le résumé des commandes Nautilus

1. `kubectl get nodes`
1. `kubectl get pods`
2. `kubectl create -f pod.yaml`
3. `kubectl top pod` or [Grafana namespace dashboards](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods.)
4. `kubectl logs bean`, additionally add the `-f` flag for stream.
    - `kubectl logs -f "$(kubectl get pods | awk 'FNR == 2 {print $1}')"`
5. `kubectl exec -it bean -- /bin/bash`
6. `kubectl delete -f pod.yaml`

## Primality Ouput

```
  7   2 0.5938 (2, 0.59375)
  8   1 0.6500 (1, 0.65)
  9  -1 0.5391 (4, 0.5390625)
 10   3 0.6488 (3, 0.6487603305785123)
 11  21 0.5766 (21, 0.5765765765765766)
 12   9 0.6193 (9, 0.619277108433735)
 13   5 0.5608 (5, 0.5608365019011406)
 14  15 0.6669 (15, 0.6668882978723404)
 15  17 0.7652 (17, 0.765183615819209)
 16  31 0.6549 (31, 0.6548746518105849)
 17  19 0.5733 (19, 0.5733163664839468)
 18  28 0.7779 (28, 0.7779294360030904)
 19  20 0.7001 (20, 0.7000836436241973)
 20  20 0.6246 (20, 0.6245861044290364)
 21  51 0.7064 (51, 0.7064156671109136)
 22  44 0.5865 (44, 0.586510292264762)
 23  53 0.7430 (53, 0.7430030618838739)
 24  -1 0.5000 (53, 0.5002392660135455)
 25  -1 0.5000 (24, 0.5000005437389009)
 26  -1 0.5000 (45, 0.5000142448334272)
```

## Different Models

The `model_con_full.p` was trained on all of 32-bit prime numbers and
therefore should be tested with them,

```python
primes_numbers = primes32bit()
network = np.load('../../saved/model_con_full.p', allow_pickle=True)
```

The The `model_con.p` was trained locally using, a subset of all
32-bit primes. And therefore should be tested as,

```python
primes_numbers = primes32bit()[:2**25]
network = np.load('../../saved/model_con.p', allow_pickle=True)
```
## Hyper-parameters of the Full Model

Following are how I trained the `model_con_full.p`.

```
activation logistic
solver adam
alpha 0.1
batch_size auto
learning_rate constant
learning_rate_init 0.001
power_t 0.5
max_iter 4294967296
loss log_loss
hidden_layer_sizes 100
shuffle True
random_state 0
tol 0.0001
verbose True
warm_start False
momentum 0.9
nesterovs_momentum True
early_stopping False
validation_fraction 0.1
beta_1 0.9
beta_2 0.999
epsilon 1e-08
n_iter_no_change 10
max_fun 15000
```

## Erratic Data

- `erratic.npy` is a sorted Numpy array of 5000 random prime numbers
  greater than 2^(50).

The erratic array was roughly generated like this (I don't remember
the exact code I ran back in summer 2021).

```
primes = []
n = 2^50
i = 0
while i < 5000:
  n = p[-1]
  primes.append(randprime(n, 2 * n + 1))
```

The `randprime` function is from the `sympy` library.

## Possibly Useful Code Snippets

Here is how the hyper-parameters of either `model_con.p` or
`model_err.p` can be printed.

```python
network = np.load('saved/model_con.p', allow_pickle=True)
params = vars(network)
for k in params:
    if not (k.startswith('_') or k.endswith('_')):
        print(k, params[k])
```

And the confusion matrix can be printed for the test labels `_y` and
predicted labels `preds`,

```python
from sklearn.metrics import confusion_matrix

print('Confusion Matrix:')
print(confusion_matrix(_y.flatten(), preds.flatten()) / total_bits)
```
