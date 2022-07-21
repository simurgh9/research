# Erratic Data

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

# Possibly Useful Code Snippets

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
