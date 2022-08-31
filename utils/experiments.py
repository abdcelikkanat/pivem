import numpy as np
from utils.common import linearIdx2matIdx, set_seed


def init_worker(param_r, param_nodes_num, param_all_events):
    global r, nodes_num, all_events

    r = param_r
    nodes_num = param_nodes_num
    all_events = param_all_events


def generate_pos_samples(seed, event_pair, events, low=0.0, high=1.0):
    global r
    set_seed(seed)

    pos_samples = [[event_pair[0], event_pair[1], max(low, e - r), min(high, e + r)] for e in events]

    return pos_samples


def generate_neg_samples(seed, e, low=0.0, high=1.0):
    global r, nodes_num, all_events
    set_seed(seed)

    valid_sample = False
    while not valid_sample:
        pairs_num = int(nodes_num * (nodes_num - 1) / 2)
        sampled_linear_pair_idx = np.random.randint(pairs_num, size=1)
        sampled_pair = linearIdx2matIdx(idx=sampled_linear_pair_idx, n=nodes_num, k=2)
        events = np.asarray(all_events[sampled_pair][1])
        # If there is no any link on the interval [e-r, e+r), add it into the negative samples
        valid_sample = True if np.sum((min(high, e + r) > events) * (events >= max(low, e - r))) == 0 else False
        if not valid_sample:
            e = np.random.uniform(low=low, high=high, size=1).tolist()[0]

    return [sampled_pair[0], sampled_pair[1], max(low, e - r), min(high, e + r)]