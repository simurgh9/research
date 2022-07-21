from net.ffnn import FFNN
from time import time as seconds  # noqa: F40
from multiprocessing import Process, Queue, active_children


class ParallelBatch(FFNN):

    def SGD(self, print_steps):
        error_sum_over_bt, q = 0, Queue()
        for bt in self.batches():
            Process(target=lambda B, q: q.put(self.gradient_bt(B)),
                    args=(
                        bt,
                        q,
                    )).start()
        while len(active_children()) > 0 or not q.empty():
            while not q.empty():
                grad_bt, error_bt = q.get()
                error_sum_over_bt += error_bt
                self.Wb = self.Wb - self.learning_rate() * grad_bt
        return error_sum_over_bt / self.total_bt()  # mean of batch errors


class ParallelExample(FFNN):

    def gradient_bt(self, mini_batch):
        g_sum, error_sum_over_x, q = 0, 0, Queue()
        for x, y in mini_batch:
            # s = seconds()
            Process(target=lambda x, y, q: q.put(self.backpropagation(x, y)),
                    args=(
                        x,
                        y,
                        q,
                    )).start()
            # print(seconds() - s)
        while len(active_children()) > 0 or not q.empty():
            while not q.empty():
                gradient, error = q.get()
                error_sum_over_x += error
                g_sum += gradient
        return g_sum / self.bt, error_sum_over_x / self.bt
