import pickle

import torch
from src.base import BaseModel
from src.learning import LearningModel
from torch.utils.data import DataLoader
from src.dataset import Dataset
import numpy as np
import pandas as pd
from src.animation import Animation
import os
import sys
import utils
import pickle as pkl


# Set some parameters
dim = 2
K = 25
bins_num = 100
prior_lambda = "1e0"
batch_size = 100
learning_rate = 0.1
epoch_num = 100
steps_per_epoch = 10
seed = 12
last_time = 0.9

###
dataset_name = f"nodes=100_cn=20_bins=10_intensity=5_seed=19"
model_name = f"dec_{dataset_name}_B={bins_num}_K={K}_lambda={prior_lambda}_dim={dim}_lt={last_time}"
model_name += f"_epoch={epoch_num}_spe={steps_per_epoch}_bs={batch_size}_lr={learning_rate}_seed={seed}"

node2group_path = os.path.join(utils.BASE_FOLDER, "experiments", "datasets", dataset_name, "node2group.pkl")
model_path = os.path.join(utils.BASE_FOLDER, "experiments", "models_mr=0.2_cr=0.0_pr=0.1", model_name+".model")
anim_path = os.path.join(utils.BASE_FOLDER, "experiments", "animations", f"{model_name}.mp4")

# Read the node2group information
if os.path.exists(node2group_path):
    with open(node2group_path, 'rb') as f:
        node2group = pickle.load(f)
else:
    node2group = None

# Load the model
assert os.path.exists(model_path), f"The model file does not exist! {model_path}"
kwargs, lm_state = torch.load(model_path, map_location=torch.device("cpu"))
kwargs["device"] = "cpu"
lm = LearningModel(**kwargs)
lm.load_state_dict(lm_state)

frame_times = torch.linspace(0, lm.get_last_time(), 100)
embs = lm.get_xt(
    events_times_list=torch.cat([frame_times]*lm.get_number_of_nodes()),
    x0=torch.repeat_interleave(lm.get_x0(), repeats=len(frame_times), dim=0),
    v=torch.repeat_interleave(lm.get_v(), repeats=len(frame_times), dim=1)
).reshape((lm.get_number_of_nodes(), len(frame_times),  lm.get_dim())).transpose(0, 1).detach().numpy()
anim = Animation(
    embs, data=(None, None),
    fps=12,
    node2color=None if node2group is None else [node2group[idx] for idx in range(lm.get_number_of_nodes())],
    frame_times=frame_times.numpy()
)
anim.save(anim_path)