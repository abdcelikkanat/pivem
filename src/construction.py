import os
import torch
import numpy as np
import pickle as pkl
from src.nhpp import NHPP
from src.base import BaseModel
import utils


class InitialPositionVelocitySampler:

    def __init__(self, dim: int, bins_num: int, cluster_sizes: list,
                 prior_lambda: float, prior_sigma: float, prior_B_x0_c: float, prior_B_ls: float,
                 device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        self.__dim = dim
        self.__bins_num = bins_num
        self.__cluster_sizes = cluster_sizes
        self.__prior_lambda = prior_lambda
        self.__prior_sigma = prior_sigma
        self.__prior_B_ls = prior_B_ls
        self.__prior_B_x0_c = prior_B_x0_c
        self.__time_interval_lengths = [1]*bins_num

        self.__nodes_num = sum(cluster_sizes)
        self.__K = len(cluster_sizes)

        self.__device = device
        self.__verbose = verbose
        self.__seed = seed

    def sample(self):

        # Get the factor of B matrix, (bins)
        bin_centers = torch.arange(0.5, 0.5*(self.__bins_num+1), 0.5).view(1, self.__bins_num)

        # Construct the factor of B matrix (bins)
        B_factor = BaseModel.get_B_factor(
            bin_centers1=bin_centers, bin_centers2=bin_centers,
            prior_B_x0_c=torch.as_tensor(self.__prior_B_x0_c, dtype=torch.float),
            prior_B_ls=torch.as_tensor(self.__prior_B_ls),
        )

        # Get the factor of C matrix, (nodes)
        prior_C_Q = torch.zeros(size=(self.__nodes_num, self.__K), dtype=torch.float)
        for k in range(self.__K):
            prior_C_Q[sum(self.__cluster_sizes[:k]):sum(self.__cluster_sizes[:k + 1]), k] = 1
        C_factor = BaseModel.get_C_factor(prior_C_Q)

        # Get the factor of D matrix, (dimension)
        D_factor = BaseModel.get_D_factor(dim=self.__dim)

        # Sample the initial position and velocity vectors
        final_dim = self.__nodes_num * (self.__bins_num+1) * self.__dim
        cov_factor = self.__prior_lambda * torch.kron(
            B_factor.contiguous(), torch.kron(C_factor, D_factor.contiguous())
        )
        cov_diag = (self.__prior_lambda ** 2) * (self.__prior_sigma ** 2) * torch.ones(final_dim)
        lmn = torch.distributions.LowRankMultivariateNormal(
            loc=torch.zeros(size=(final_dim,)),
            cov_factor=cov_factor,
            cov_diag=cov_diag
        )

        sample = lmn.sample().reshape(shape=(self.__bins_num + 1, self.__nodes_num, self.__dim))

        x0, v = torch.split(sample, [1, self.__bins_num])
        x0 = x0.squeeze(0)

        return x0, v, (1.0 * self.__bins_num)


class ConstructionModel(BaseModel):

    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, last_time: float,
                 bins_num: int, device: torch.device = "cpu", verbose: bool = False, seed: int = 0):

        super(ConstructionModel, self).__init__(
            x0=x0,
            v=v,
            beta=beta,
            last_time=last_time,
            bins_num=bins_num,
            device=device,
            verbose=verbose,
            seed=seed
        )

        self.__events = self.__sample_events()

    def __get_critical_points(self, i: int, j: int, x: torch.tensor):

        bin_bounds = self.get_bins_bounds()

        # Add the initial time point
        critical_points = []

        for idx in range(self.get_bins_num()):

            interval_init_time = bin_bounds[idx]
            interval_last_time = bin_bounds[idx+1]

            # Add the initial time point of the interval
            critical_points.append(interval_init_time)

            # Get the differences
            delta_idx_x = x[idx, i, :] - x[idx, j, :]
            delta_idx_v = self.get_v()[idx, i, :] - self.get_v()[idx, j, :]

            # For the model containing only position and velocity
            # Find the point in which the derivative equal to 0
            t = - np.dot(delta_idx_x, delta_idx_v) / (np.dot(delta_idx_v, delta_idx_v) + utils.EPS) + interval_init_time

            if interval_init_time < t < interval_last_time:
                critical_points.append(t)

        # Add the last time point
        critical_points.append(bin_bounds[-1])

        return critical_points

    def __sample_events(self, nodes: torch.tensor = None) -> dict:

        if nodes is not None:
            raise NotImplementedError("It must be implemented for given specific nodes!")

        node_pairs = torch.triu_indices(
            row=self.get_number_of_nodes(), col=self.get_number_of_nodes(), offset=1, device=self.get_device()
        )

        # Upper triangular matrix of lists
        events_time = {
            i: {j: [] for j in range(i+1, self.get_number_of_nodes())} for i in range(self.get_number_of_nodes()-1)
        }
        # Get the positions at the beginning of each time bin for every node
        x = self.get_xt(
            events_times_list=self.get_bins_bounds()[:-1].repeat(self.get_number_of_nodes(), ),
            x0=torch.repeat_interleave(self.get_x0(), repeats=self.get_bins_num(), dim=0),
            v=torch.repeat_interleave(self.get_v(), repeats=self.get_bins_num(), dim=1)
        ).reshape((self.get_number_of_nodes(), self.get_bins_num(),  self.get_dim())).transpose(0, 1)

        for i, j in zip(node_pairs[0], node_pairs[1]):
            # Define the intensity function for each node pair (i,j)
            intensity_func = lambda t: self.get_intensity(
                times_list=torch.as_tensor([t]), node_pairs=torch.as_tensor([[i], [j]])
            ).item()
            # Get the critical points
            critical_points = self.__get_critical_points(i=i, j=j, x=x)
            # Simulate the src
            nhpp_ij = NHPP(
                intensity_func=intensity_func, critical_points=critical_points,
                seed=self.get_seed() + i * self.get_number_of_nodes() + j
            )
            ij_events_time = nhpp_ij.simulate()
            # Add the event times
            events_time[i.item()][j.item()].extend(ij_events_time)

        return events_time

    def get_events(self):

        return self.__events

    def save(self, folder_path):
        events, pairs = [], []
        for i, j in utils.pair_iter(n=self.get_number_of_nodes()):
            pair_events = self.__events[i][j]
            if len(pair_events):
                pairs.append([i, j])
                events.append(pair_events)

        with open(os.path.join(folder_path, "pairs.pkl"), 'wb') as f:
            pkl.dump(pairs, f)

        with open(os.path.join(folder_path, "events.pkl"), 'wb') as f:
            pkl.dump(events, f)