import os
import sys
import pickle
from argparse import ArgumentParser, RawTextHelpFormatter
from src.dataset import Dataset
from utils.common import set_seed
import numpy as np
from sklearn.utils import shuffle

########################################################################################################################

parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--input_folder', type=str, required=True, help='Path of the input dataset folder'
)
parser.add_argument(
    '--output_folder', type=str, required=True, help='Path of the output dataset folder'
)
parser.add_argument(
    '--log_file', type=str, required=False, default="", help='Path for the log file'
)
parser.add_argument(
    '--pr', type=float, required=False, default=0.05, help='Prediction ratio'
)
parser.add_argument(
    '--mr', type=float, required=False, default=0.025, help='Masking ratio'
)
parser.add_argument(
    '--cr', type=float, required=False, default=0.025, help='Completion ratio'
)
parser.add_argument(
    '--trials_num', type=int, required=False, default=10, help='Number of trials'
)
parser.add_argument(
    '--verbose', type=bool, required=False, default=True, help='Verbose'
)
parser.add_argument(
    '--seed', type=int, required=False, default=19, help='Seed value'
)
args = parser.parse_args()

########################################################################################################################

# Set some parameters
input_folder = args.input_folder
output_folder = args.output_folder
log_file = args.log_file
prediction_ratio = args.pr
masking_ratio = args.mr
completion_ratio = args.cr
trials_num = args.trials_num
verbose = args.verbose
seed = args.seed
# Set the seed value
set_seed(seed=seed)

# Create the target folder
os.makedirs(output_folder)
if log_file != "":
    orig_stdout = sys.stdout
    f = open(log_file, 'w')
    sys.stdout = f


########################################################################################################################
# Sample the pairs for masking and the completion experiment

dataset = Dataset(path=input_folder, normalize=True, seed=seed)
nodes_num = dataset.number_of_nodes()
pairs, events = dataset.get_pairs(), dataset.get_events()

# We assume that sample size is even number
completion_size = int(dataset.number_of_event_pairs() * completion_ratio)
completion_size = completion_size if completion_size % 2 == 0 else completion_size + 1
mask_size = int(dataset.number_of_event_pairs() * masking_ratio)
mask_size = mask_size if mask_size % 2 == 0 else mask_size + 1
if verbose:
    print("- For the completion experiment, pair sampling is being performed...")
residual_pairs, residual_events = pairs.copy(), events.copy()
for trial_idx in range(1, trials_num+1):

    residual_pairs, residual_events = shuffle(residual_pairs, residual_events)
    # Keep the number of nodes in the residual network fixed
    # Size is (2 x sample_size) since we also need mask pairs for validation
    if len(np.unique(residual_pairs[:-(completion_size+mask_size)])) == nodes_num:
        print(f"\t+ Successful! (Trial:{trial_idx}/{trials_num})")
        break
    else:
        if verbose:
            print(f"\t+ Unsuccessful: The residual network has less number of nodes. (Trial:{trial_idx}/{trials_num})")

assert trial_idx <= trials_num, "\t+ Completion pairs could not be generated! Please choose a smaller ratio!"

residual_pairs, completion_pairs = residual_pairs[:-completion_size], residual_pairs[-completion_size:]
residual_events, completion_events = residual_events[:-completion_size], residual_events[-completion_size:]

masked_pairs = residual_pairs[-mask_size:]
masked_events = residual_events[-mask_size:]  # In fact, we don't need to store the masked events

########################################################################################################################
# Split the residual network and completion pairs

assert 0 <= prediction_ratio < 1.0, "Prediction ratio must be smaller than 1.0!"
split_time = 1.0 - prediction_ratio

train_pairs, train_events = [], []
pred_pairs, pred_events = [], []
for pair, pair_events in zip(residual_pairs, residual_events):

    train_pairs.append(pair), train_events.append([])
    pred_pairs.append(pair), pred_events.append([])

    for e in pair_events:
        if e <= split_time:
            train_events[-1].append(e)
        else:
            pred_events[-1].append(e)

    if len(train_events[-1]) == 0:
        train_pairs.pop()
        train_events.pop()

    if len(pred_events[-1]) == 0:
        pred_pairs.pop()
        pred_events.pop()

assert len(np.unique(train_pairs)) == nodes_num, \
    "Training set contains less number of nodes, please choose different prediction ratio!"

# Remove the event times greater than the split ratio in the testing and masking events
idx = 0
while idx < len(completion_events):
    completion_events[idx] = [e for e in completion_events[idx] if e <= split_time]
    if len(masked_events[idx]) == 0:
        completion_pairs.pop(idx)
        completion_events.pop(idx)
    else:
        idx += 1
idx = 0
while idx < len(masked_events):
    masked_events[idx] = [e for e in masked_events[idx] if e <= split_time]
    if len(masked_events[idx]) == 0:
        masked_pairs.pop(idx)
        masked_events.pop(idx)
    else:
        idx += 1

########################################################################################################################

if verbose:
    print("- All the processes have been completed!")
    print(f"\t+ The training set has {len(train_pairs)} pairs.")
    all_events = [e for te in train_events for e in te]
    print(f"\t\t* The min/max events in training set are {min(all_events)}/{max(all_events)}.")
    print(f"\t+ The completion set has {len(completion_pairs)} pairs.")
    all_events = [e for te in completion_events for e in te]
    print(f"\t\t* The min/max events in completion set are {min(all_events)}/{max(all_events)}.")
    print(f"\t+ The masking set has {len(masked_pairs)} pairs.")
    all_events = [e for te in masked_events for e in te]
    print(f"\t\t* The min/max events in masking set are {min(all_events)}/{max(all_events)}.")
    if completion_size != len(completion_pairs):
        print("\t+ {} pairs have been removed due to the prediction set (Desired: {}).".format(
            completion_size-len(completion_pairs), completion_size)
        )
    if mask_size != len(masked_pairs):
        print("\t+ {} pairs have been removed due to the prediction set (Desired: {}).".format(
            mask_size-len(masked_pairs), mask_size)
        )

########################################################################################################################

if verbose:
    print("- The files are being written...")

# Save the residual pair and events
with open(os.path.join(output_folder, "pairs.pkl"), 'wb') as f:
    pickle.dump(train_pairs, f)
with open(os.path.join(output_folder, "events.pkl"), 'wb') as f:
    pickle.dump(train_events, f)
if dataset.get_groups() is not None:
    with open(os.path.join(output_folder, "node2group.pkl"), 'wb') as f:
        pickle.dump(dataset.get_groups(), f)

# Save the completion pairs
os.makedirs(os.path.join(output_folder, "completion"))
with open(os.path.join(output_folder, "completion", "pairs.pkl"), 'wb') as f:
    pickle.dump(masked_pairs, f)
with open(os.path.join(output_folder, "completion", "events.pkl"), 'wb') as f:
    pickle.dump(masked_events, f)

# Save the mask pairs
os.makedirs(os.path.join(output_folder, "mask"))
with open(os.path.join(output_folder, "mask", "pairs.pkl"), 'wb') as f:
    pickle.dump(masked_pairs, f)
with open(os.path.join(output_folder, "mask", "events.pkl"), 'wb') as f:
    pickle.dump(masked_events, f)

# Save the mask pairs
os.makedirs(os.path.join(output_folder, "prediction"))
with open(os.path.join(output_folder, "prediction", "pairs.pkl"), 'wb') as f:
    pickle.dump(masked_pairs, f)
with open(os.path.join(output_folder, "prediction", "events.pkl"), 'wb') as f:
    pickle.dump(masked_events, f)

if verbose:
    print(f"\t+ Completed.")


if log_file != "":
    sys.stdout = orig_stdout
    f.close()