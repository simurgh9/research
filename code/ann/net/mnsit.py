from os import makedirs
from gzip import open as gzip_open
from urllib.error import HTTPError
from urllib.request import urlopen, Request
from os.path import join as path_join, exists as path_exists

import numpy as np
import matplotlib.pyplot as plt

plt.tight_layout()


class MNSIT:
    DOWNLOAD_ADDRESS = 'http://yann.lecun.com/exdb/mnist/'
    # training set images (9912422 bytes)
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    # training set labels (28881 bytes)
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    # test set images (1648877 bytes)
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    # test set labels (4542 bytes)
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    def __init__(self, path=None):
        self.files = [
            self.TRAIN_IMAGES, self.TEST_IMAGES, self.TRAIN_LABELS,
            self.TEST_LABELS
        ]
        self.path = path
        if self.path is None:
            self.path = path_join('.', 'mnsit_data', '')
        if not path_exists(self.path):
            makedirs(self.path)
            self.download()
        self.x, self.tx, self.y, self.ty = self.load()

    def get_data(self):
        return self.x, self.tx, self.y, self.ty

    def plot_image(self, i, source='training'):
        if source == 'training':
            if i > self.x.shape[0]:
                raise IndexError(
                    f'Index {i} out of bounds in {self.x.shape[0]} examples.')
            image = np.array(self.x[i], dtype='float')
            plt.imshow(image, cmap='gray_r')
            plt.title(f'Label: {self.y[i]}')
            return plt.show()
        elif source == 'testing':
            if i > self.tx.shape[0]:
                raise IndexError(
                    f'Index {i} out of bounds in {self.tx.shape[0]} examples.')
            image = np.array(self.tx[i], dtype='float')
            plt.imshow(image, cmap='gray')
            plt.title('Label: {self.ty[i]}')
            return plt.show()
        else:
            raise ValueError('Source must either be "testing" or "training".')

    def download(self, path=None):
        path = self.path if path is None else path
        for fl_name in self.files:
            responce_bytes = self.http_get(self.DOWNLOAD_ADDRESS + fl_name)
            with open(path + fl_name, 'wb') as f:
                f.write(responce_bytes)

    def load(self, path=None):
        path = self.path if path is None else path
        ret = [None] * 4
        for i, fl_name in enumerate(self.files):
            with gzip_open(path + fl_name, 'rb') as f:
                arr = bytearray(f.read())
            # magic_number = int.from_bytes(arr[:4], byteorder='big')
            num_examples = int.from_bytes(arr[4:8], byteorder='big')
            if i < 2:
                rows = int.from_bytes(arr[8:12], byteorder='big')
                cols = int.from_bytes(arr[12:16], byteorder='big')
                ret[i] = np.array(arr[16:]).reshape(num_examples, rows, cols)
            else:
                ret[i] = np.array(arr[8:])
        return tuple(ret)

    def http_get(self, url):
        USER_AGENT = ('Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17'
                      '(KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17')
        response = urlopen(Request(url, headers={'User-Agent': USER_AGENT}))
        if response.status != 200:
            raise HTTPError(f'HTTP Request failed with code {response.status}')
        return response.read()
