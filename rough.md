## Le résumé des commandes Nautilus

1. `kubectl get nodes`
2. `kubectl create -f pod.yaml`
3. `kubectl top pod` or [Grafana namespace dashboards](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods.)
4. `kubectl logs bean`, additionally add the `-f` flag for stream.
    - `kubectl logs -f "$(kubectl get pods | awk 'FNR == 2 {print $1}')"`
5. `kubectl exec -it bean -- /bin/bash`
6. `kubectl delete -f pod.yaml`

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
network = np.load('../../saved/model_con_full.p', allow_pickle=True)
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
