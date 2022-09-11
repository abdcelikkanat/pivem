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
    '--interval', type=float, required=False, default=0.001, help='Half length of the interval'
)
parser.add_argument(
    '--russ', type=int, required=False, default=10000, help='reconstruction upper sample size limit'
)
parser.add_argument(
    '--last_time', type=float, required=False, default=0.9, help='Last training time'
)
parser.add_argument(
    '--seed', type=int, required=False, default=19, help='Seed value'
)
args = parser.parse_args()

########################################################################################################################

samples_folder = args.samples_folder
last_time = args.last_time
event_interval = args.interval
russ = args.russ  # reconstruction_upper_sample_size
seed = args.seed

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
last_time = last_time
train_dataset = Dataset(data=(train_events, train_pairs, range(nodes_num)), normalize=False, verbose=False, seed=seed)


########################################################################################################################

# Network reconstruction experiment
print("Reconstruction Experiment")

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

# Network completion experiment
print("Completion Experiment")

# Read the completion pairs and events
completion_folder = os.path.join(samples_folder, "completion")
with open(os.path.join(completion_folder, "pairs.pkl"), 'rb') as f:
    completion_pairs = pickle.load(f)
with open(os.path.join(completion_folder, "events.pkl"), 'rb') as f:
    completion_events = pickle.load(f)

# Construct positive samples
completion_pos_samples = [
    [
        completion_pairs[idx][0], completion_pairs[idx][1],
        max(0, completion_events[idx]-event_interval),
        min(last_time, completion_events[idx]+event_interval)
    ]
    for idx in range(len(completion_pairs))
]

# Construct the negative samples
completion_neg_samples = []
neg_events = (np.random.rand(len(completion_pos_samples)) * last_time).tolist()
neg_linear_indices = np.random.choice(nodes_num*(nodes_num-1)//2, size=(len(completion_pos_samples), ), replace=True)
count = 0
while count < len(completion_pos_samples):

    candidate_linear_idx = np.random.choice(nodes_num * (nodes_num - 1) // 2, size=(1,), replace=True)[0]
    candidate_pair = linearIdx2matIdx(idx=candidate_linear_idx, n=nodes_num, k=2)
    candidate_e = (np.random.rand() * last_time)

    valid_sample = True
    for e in train_dataset[candidate_pair][1]:
        if max(0, candidate_e - event_interval) <= e <= min(last_time, candidate_e + event_interval):
            valid_sample = False
            break

    if valid_sample:
        completion_neg_samples.append([
            candidate_pair[0], candidate_pair[1],
            max(0, candidate_e - event_interval),
            min(last_time, candidate_e + event_interval)
        ])

        count += 1

########################################################################################################################

# Network prediction experiment
print("Prediction Experiment")

# Read the completion pairs and events
prediction_folder = os.path.join(samples_folder, "prediction")
with open(os.path.join(prediction_folder, "pairs.pkl"), 'rb') as f:
    prediction_pairs = pickle.load(f)
with open(os.path.join(prediction_folder, "events.pkl"), 'rb') as f:
    prediction_events = pickle.load(f)

# Prediction dataset
prediction_dataset = Dataset(
    data=(prediction_events, prediction_pairs, range(nodes_num)), normalize=False, verbose=False, seed=seed
)

all_prediction_pairs, all_prediction_events = [], []
for pair, pair_events in zip(prediction_pairs, prediction_events):
    for e in pair_events:
        all_prediction_pairs.append(pair)
        all_prediction_events.append(e)

# Construct positive samples
if russ < len(all_prediction_pairs):
    chosen_indices = np.random.choice(a=range(len(all_prediction_pairs)), size=(russ, ), replace=False)
    prediction_pos_samples = [
        [
            all_prediction_pairs[idx][0], all_prediction_pairs[idx][1],
            max(last_time, all_prediction_events[idx]-event_interval),
            min(1.0, all_prediction_events[idx]+event_interval)
        ]
        for idx in chosen_indices
    ]
else:
    prediction_pos_samples = [
        [pair[0], pair[1], max(last_time, e-event_interval), min(1.0, e+event_interval)]
        for pair, e in zip(all_prediction_pairs, all_prediction_events)
    ]

# Construct the negative samples
prediction_neg_samples = []
neg_events = (np.random.rand(len(prediction_pos_samples)) * (1.0 - last_time) + last_time).tolist()
neg_linear_indices = np.random.choice(nodes_num*(nodes_num-1)//2, size=(len(prediction_pos_samples), ), replace=True)
count = 0
while count < len(prediction_pos_samples):

    candidate_linear_idx = np.random.choice(nodes_num * (nodes_num - 1) // 2, size=(1,), replace=True)[0]
    candidate_pair = linearIdx2matIdx(idx=candidate_linear_idx, n=nodes_num, k=2)
    candidate_e = (np.random.rand() * (1.0 - last_time) + last_time)

    valid_sample = True
    for e in prediction_dataset[candidate_pair][1]:
        if max(last_time, candidate_e - event_interval) <= e <= min(1.0, candidate_e + event_interval):
            valid_sample = False
            break

    if valid_sample:
        prediction_neg_samples.append([
            candidate_pair[0], candidate_pair[1],
            max(last_time, candidate_e - event_interval),
            min(1.0, candidate_e + event_interval)
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

# Save the completion samples
completion_path = os.path.join(samples_folder, f"completion_interval={event_interval}")
os.makedirs(completion_path)
with open(os.path.join(completion_path, "pos.samples"), 'wb') as f:
    pickle.dump(completion_pos_samples, f)
with open(os.path.join(completion_path, "neg.samples"), 'wb') as f:
    pickle.dump(completion_neg_samples, f)

# Save the completion samples
prediction_path = os.path.join(samples_folder, f"prediction_russ={russ}_interval={event_interval}")
os.makedirs(prediction_path)
with open(os.path.join(prediction_path, "pos.samples"), 'wb') as f:
    pickle.dump(prediction_pos_samples, f)
with open(os.path.join(prediction_path, "neg.samples"), 'wb') as f:
    pickle.dump(prediction_neg_samples, f)

########################################################################################################################

