import os
from argparse import ArgumentParser, RawTextHelpFormatter
from utils.common import set_seed, linearIdx2matIdx
import numpy as np
from src.dataset import Dataset
import pickle


########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--samples_folder', type=str, required=True, help='Path of the samples folder'
)
parser.add_argument(
    '--interval', type=float, required=True, default=0.001, help='Half length of the interval'
)
parser.add_argument(
    '--russ', type=int, required=True, default=10000, help='reconstruction upper sample size limit'
)
parser.add_argument(
    '--pr', type=float, required=False, default=0.1, help='Prediction ratio'
)
parser.add_argument(
    '--mr', type=float, required=False, default=0.2, help='Masking ratio'
)
parser.add_argument(
    '--cr', type=float, required=False, default=0, help='Completion ratio'
)
parser.add_argument(
    '--seed', type=int, required=False, default=19, help='Seed value'
)
args = parser.parse_args()

########################################################################################################################

event_interval = args.interval
russ = args.russ  # reconstruction_upper_sample_size
seed = args.seed

# Sample folder settings
mask_ratio = args.mr
completion_ratio = args.cr
prediction_ratio = args.pr

samples_folder = args.samples_folder

########################################################################################################################

# Set the seed value
set_seed(seed=seed)

########################################################################################################################

# Read the train pairs and events
train_folder = os.path.join(samples_folder, "train")
with open(os.path.join(train_folder, "pairs.pkl"), 'rb') as f:
    train_pairs = pickle.load(f)
with open(os.path.join(train_folder, "events.pkl"), 'rb') as f:
    train_events = pickle.load(f)

# Construct train dataset
nodes_num = len(np.unique(train_pairs))
last_time = 1.0 - prediction_ratio
train_dataset = Dataset(data=(train_events, train_pairs, range(nodes_num)), normalize=False, verbose=False, seed=seed)


########################################################################################################################

# Network reconstruction experiment
all_pairs, all_events = [], []
for pair, pair_events in zip(train_pairs, train_events):
    for e in pair_events:
        all_pairs.append(pair)
        all_events.append(e)

# Construct positive samples
if russ < len(all_pairs):
    chosen_indices = np.random.choice(a=range(len(all_pairs)), size=(russ, ), replace=False)
    pos_samples = [
        [
            all_pairs[idx][0], all_pairs[idx][1],
            max(0, all_events[idx]-event_interval),
            min(last_time, all_events[idx]+event_interval)
        ]
        for idx in chosen_indices
    ]
else:
    pos_samples = [
        [pair[0], pair[1], max(0, e-event_interval), min(last_time, e+event_interval)]
        for pair, e in zip(all_pairs, all_events)
    ]

# Construct the negative samples
neg_samples = []
neg_events = (np.random.rand(len(pos_samples)) * last_time).tolist()
neg_linear_indices = np.random.choice(nodes_num*(nodes_num-1)//2, size=(len(pos_samples), ), replace=True)
count = 0
while count < len(pos_samples):

    candidate_linear_idx = np.random.choice(nodes_num * (nodes_num - 1) // 2, size=(1,), replace=True)[0]
    candidate_pair = linearIdx2matIdx(idx=candidate_linear_idx, n=nodes_num, k=2)
    candidate_e = (np.random.rand() * last_time)

    valid_sample = True
    for e in train_dataset[candidate_pair][1]:
        if max(0, candidate_e - event_interval) <= e <= min(last_time, candidate_e + event_interval):
            valid_sample = False
            break

    if valid_sample:
        neg_samples.append([
            candidate_pair[0], candidate_pair[1],
            max(0, candidate_e - event_interval),
            min(last_time, candidate_e + event_interval)
        ])

        count += 1

########################################################################################################################

# Save the reconstruction samples
reconstruction_path = os.path.join(samples_folder, f"reconstruction_russ={russ}_interval={event_interval}")
os.makedirs(reconstruction_path)
with open(os.path.join(reconstruction_path, "pos.samples"), 'wb') as f:
    pickle.dump(pos_samples, f)
with open(os.path.join(reconstruction_path, "neg.samples"), 'wb') as f:
    pickle.dump(neg_samples, f)

########################################################################################################################

