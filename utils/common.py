import os
import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Path definitions
BASE_FOLDER = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

# Constants
EPS = 1e-6
INF = 1e+6
PI = math.pi
LOG2PI = math.log(2*PI)


def str2int(text):

    return int(sum(map(ord, text)) % 1e6)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pair_iter(n, undirected=True):

    if undirected:
        for i in range(n):
            for j in range(i+1, n):
                yield i, j

    else:

        for i in range(n):
            for j in range(i+1, n):
                yield i, j


def pairIdx2flatIdx(i, j, n, undirected=True):

    if undirected:

        return (n-1) * i - int(i*(i+1)/2) + (j-1)

    else:

        return i*n + j


def linearIdx2matIdx(idx, n, k):

    result = np.arange(k)

    num = 0
    while num < idx:

        col_idx = k-1
        result[col_idx] += 1
        while result[col_idx] == n+(col_idx-k+1) :
            col_idx -= 1
            result[col_idx] += 1

        if col_idx < k-1:
            while col_idx < k-1:
                result[col_idx+1] = result[col_idx] + 1
                col_idx += 1

        num += 1

    return result.tolist()


def plot_events(num_of_nodes, samples, labels, title=""):

    def node_pairs(num_of_nodes):
        for idx1 in range(num_of_nodes):
            for idx2 in range(idx1 + 1, num_of_nodes):
                yield idx1, idx2
    pair2idx = {pair: idx for idx, pair in enumerate(node_pairs(num_of_nodes))}

    samples, labels = shuffle(samples, labels)

    plt.figure(figsize=(18, 10))
    x = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}
    y = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}
    c = {i: {j: [] for j in range(i+1, num_of_nodes)} for i in range(num_of_nodes)}

    for sample, label in zip(samples, labels):

        idx1, idx2, e = int(sample[0]), int(sample[1]), float(sample[2])

        x[idx1][idx2].append(e)
        y[idx1][idx2].append(pair2idx[(idx1, idx2)])
        c[idx1][idx2].append(label)

    colors = ['.r', 'xk']
    for idx1, idx2 in node_pairs(num_of_nodes):

        for idx3 in range(len(x[idx1][idx2])):
            # if colors[c[idx1][idx2][idx3]] != '.r':
            plt.plot(x[idx1][idx2][idx3], y[idx1][idx2][idx3], colors[c[idx1][idx2][idx3]])

    plt.grid(axis='x')
    plt.title(title)
    plt.show()