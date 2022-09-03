import os
import torch
import pickle as pkl
import utils
from src.base import BaseModel
from src.construction import ConstructionModel, InitialPositionVelocitySampler
from src.animation import Animation
from src.dataset import Dataset

import math

########################################################################################################################
# Definition of the model parameters
dim = 2
beta = 0  #0.025
cluster_num = 8 #20
cluster_size = 5
nodes_num = cluster_num * cluster_size
bins_num = 100

prior_lambda = 1e0 #1e0 # scaling parameter #1e0
prior_sigma = 7e-3 #25e-3  #0.025 #0.1
prior_B_x0_c = 5e+0 #1e+1 # controls the initial node positions #2e+0
prior_B_ls = 1e-4 #1e-4 # controls how crazy the nodes are

# Set the parameters
verbose = True
seed = 19
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################################

# Definition of the folder and file paths
dataset_name = f"beta={beta}_cn={cluster_num}_cs={cluster_size}_sigma={prior_sigma}_lambda={prior_lambda}" \
               f"_prior-B-x0-c={prior_B_x0_c}_prior-B-ls={prior_B_ls}"

dataset_folder = os.path.join(utils.BASE_FOLDER, "datasets", dataset_name)
node2group_path = os.path.join(dataset_folder, "node2group.pkl")

# Check if the file exists
assert not os.path.exists(dataset_folder), "This dataset exists!"
# Create the folder of the dataset
os.makedirs(dataset_folder)

########################################################################################################################

# Sample the initial position and velocities
pvs = InitialPositionVelocitySampler(
    dim=dim, bins_num=bins_num, cluster_sizes=[cluster_size]*cluster_num,
    prior_lambda=prior_lambda, prior_sigma=prior_sigma, prior_B_x0_c=prior_B_x0_c, prior_B_ls=prior_B_ls,
    device=device, verbose=verbose, seed=seed
)
x0, v, last_time = pvs.sample()

# Construct the artificial network and save
cm = ConstructionModel(
    x0=x0, v=v, beta=torch.as_tensor([beta]*nodes_num), bins_num=bins_num, last_time=last_time,
    device=device, verbose=verbose, seed=seed
)
# Save the dataset files
cm.save(dataset_folder, normalize=True)

########################################################################################################################

# Group labels
node2group_data = {
    "node2group": {node: node // cluster_size for node in range(nodes_num)},
    "group2node": {node // cluster_size: node for node in range(nodes_num)}
}
with open(node2group_path, "wb") as f:
    pkl.dump(node2group_data, f)

########################################################################################################################

# Define the base model and save it
bm = BaseModel(x0=x0, v=v, beta=torch.as_tensor(beta), last_time=last_time, bins_num=bins_num)
torch.save(bm.state_dict(), os.path.join(dataset_folder, "bm.model"))

########################################################################################################################

# Ground truth animation
frame_times = torch.linspace(0, last_time, 100)
embs_pred = bm.get_xt(
    events_times_list=torch.cat([frame_times]*bm.get_number_of_nodes()),
    x0=torch.repeat_interleave(bm.get_x0(), repeats=len(frame_times), dim=0),
    v=torch.repeat_interleave(bm.get_v(), repeats=len(frame_times), dim=1)
).reshape((bm.get_number_of_nodes(), len(frame_times),  bm.get_dim())).transpose(0, 1).detach().numpy()

dataset = Dataset(path=dataset_folder, seed=seed)
pairs, events = dataset.get_pairs(), dataset.get_events()

# Read the group information
with open(node2group_path, "rb") as f:
    node2group_data = pkl.load(f)
node2group, group2node = node2group_data["node2group"], node2group_data["group2node"]
# Animate
node2color = [node2group[idx] for idx in range(nodes_num)]
anim = Animation(embs_pred, data=(pairs, events), fps=12, node2color=node2color, frame_times=frame_times.numpy())
anim.save(os.path.join(dataset_folder, "ground_truth_animation.mp4"))

########################################################################################################################

with open(os.path.join(dataset_folder, "info.txt"), 'w') as f:
    f.write(f"Number of nodes: {dataset.number_of_nodes()}\n")
    f.write(f"Number of event pairs: {dataset.number_of_event_pairs()}\n")
    f.write(f"Number of total events: {dataset.number_of_event_pairs()}\n")
    f.write(f"Minimum event time: {dataset.get_min_event_time()}\n")
    f.write(f"Maximum event time: {dataset.get_max_event_time()}\n")

########################################################################################################################