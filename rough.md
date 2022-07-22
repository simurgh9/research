## Summary

1. `kubectl get nodes`
2. `kubectl create -f pod.yaml`
3. `kubectl top pod` or [Grafana namespace dashboards](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods.)
4. `kubectl logs bean`, additionally add the `-f` flag for stream.
    - `kubectl logs -f "$(kubectl get pods | awk 'FNR == 2 {print $1}')"`
5. `kubectl exec -it bean -- /bin/bash`
6. `kubectl delete -f pod.yaml`


## Interaction in Nautilus

The simplest way is to create a [Pod](https://kubernetes.io/docs/concepts/workloads/pods/) within the namespace one is a
part of. Before anything, make sure you have [set-up](https://ucsd-prp.gitlab.io/userdocs/start/quickstart/)
`kubectl`. This means getting the configurations file off the Nautilus
[website](https://portal.nrp-nautilus.io/) and installing the tool itself etc. If you've done that
correctly, `kubectl get nodes` should display a list of nodes.

Here is the shortest path to a machine on Nautilus.

1. Write a `pod.yaml`,

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: bean
spec:
  containers:
  - name: bean
    image: ubuntu:latest
    resources:
      limits:
        memory: 128Mi
        cpu: 1
    command: ["sh", "-c", "apt update -y && apt install -y neofetch emacs-nox && sleep 1h"]
```

We are asking for 128 megabytes of [memory](https://kubernetes.io/docs/tasks/configure-pod-container/assign-memory-resource/#specify-a-memory-request-and-a-memory-limit) with 1
[cpu-cores](https://kubernetes.io/docs/tasks/configure-pod-container/assign-cpu-resource/#specify-a-cpu-request-and-a-cpu-limit). Note that here the sleep command is fine since this is
not a job. It is the sleep command in jobs that gets you banned.

2. Run `kubectl create -f pod.yaml`. This will create a container
   called _bean_ on the pod.

3. Check `kubectl get pods`. You should see the newly created bean in
   the list.

```
NAME   READY   STATUS    RESTARTS   AGE
bean   1/1     Running   0          8s
```

4. Check `kubectl top pod` to see the resource in use by bean. Another
   more extravagant way to check resources under utilisation by your
   pod is to visit: https://grafana.nrp-nautilus.io

The documentation also links to [Grafana namespace dashboards](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods.)

```
NAME   CPU(cores)   MEMORY(bytes)
bean   58m          319Mi
```

5. Command `kubectl logs bean` will show you the standard out and
   error from bean. And, `kubectl logs -f bean` with the `-f` flag
   will show it continuously.

6. Login with `kubectl exec -it bean -- /bin/bash`. Run `neofetch` if
   you want and `exit` to get out.

7. Delete the pod when done with `kubectl delete -f
   pod.yaml`. Nautilus's [quick start guide](https://ucsd-prp.gitlab.io/userdocs/start/quickstart/) threatens,

> Whatever you do, NEVER FORCE DELETE PODS

Ergo, please don't. Note that the `-f` flag in the above delete
command is for file and not force.

## Jobs in Nautilus

> We highly recommend using Jobs for any kind of development and
> computations in our cluster.

Above quote is from the [nautilus documentation](https://ucsd-prp.gitlab.io/userdocs/running/jobs/). A job is ran with
the similar steps we followed to get a pod but this time the YAML
looks slightly different. Sending off the yaml, getting pods (running
the job) in the namespace, checking resources etc. is all done with
the same command. Jobs are preferred because they only run as long as
the job's program is running hence no resources are ever wasted.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: bean
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: bean
        image: ubuntu:latest
        command: ["sh", "-c", "echo Results: 42"]
        resources:
          limits:
            memory: 256Mi
            cpu: 2
          requests:
            memory: 64Mi
            cpu: 1
```

---
## Link Half Gaps

https://www.dropbox.com/s/msb71j7eqevo0d7/half_gaps.bin  
https://tinyurl.com/23joktq5

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
