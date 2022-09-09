import utils
import torch
import numpy as np
import random


class BaseModel(torch.nn.Module):
    '''
    Description
    '''
    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, bins_num: int = 100,
                 last_time: float = 1.0, prior_lambda: float = 1e5, prior_sigma: torch.Tensor = None,
                 prior_B_x0_c: torch.Tensor = None, prior_B_ls: torch.Tensor = None, prior_C_Q: torch.Tensor= None,
                 device: torch.device = "cpu", verbose: bool = False, seed: int = 0, **kwargs):

        super(BaseModel, self).__init__()

        self.__x0 = x0
        self.__v = v
        self.__beta = beta
        self.__seed = seed
        self.__init_time = 0  # It is always assumed that the initial time is 0
        self.__last_time = last_time
        self.__bins_num = bins_num
        self.__verbose = verbose
        self.__device = device

        # Extract the number of nodes, dimension size and bin width
        self.__nodes_num = self.__x0.shape[0]
        self.__dim = self.__x0.shape[1]
        self.__bin_width = torch.tensor((self.__last_time - self.__init_time) / float(self.__bins_num), device=device)

        # Initialize the parameters of prior function
        # scaling factor of covariance matrix
        if prior_lambda is not None:
            self.__prior_lambda = torch.as_tensor(prior_lambda, dtype=torch.float, device=self.__device)
        # noise deviation
        if prior_sigma is not None:
            self.__prior_sigma = torch.as_tensor(prior_sigma, dtype=torch.float, device=self.__device)
        # length-scale parameter of RBF kernel used in the construction of B
        if prior_B_ls is not None:
            self.__prior_B_ls = torch.as_tensor(prior_B_ls, dtype=torch.float, device=self.__device)
        if prior_B_x0_c is not None:
            self.__prior_B_x0_c = torch.as_tensor(prior_B_x0_c, dtype=torch.float, device=self.__device)
            if self.__prior_B_x0_c.dim == 1:
                self.__prior_B_x0_c.unsquueze(0)
        if prior_C_Q is not None:
            self.__prior_C_Q = prior_C_Q  # the parameter required for the construction of the matrix C
        self.__R, self.__R_factor, self.__R_factor_inv = None, None, None  # Capacitance matrix

        # Set the seed value for reproducibility
        self.set_seed(self.__seed)

    def set_seed(self, seed=None):

        if seed is not None:
            self.__seed = seed

        random.seed(self.__seed)
        np.random.seed(self.__seed)
        torch.manual_seed(self.__seed)

    def get_x0(self, standardize=True):

        if standardize:
            return utils.standardize(self.__x0)

        else:
            return self.__x0

    def get_v(self, standardize=True):

        if standardize:
            return utils.standardize(self.__v)
        else:
            return self.__v

    def get_beta(self):

        return self.__beta

    def get_dim(self):

        return self.__dim

    def get_number_of_nodes(self):

        return self.__nodes_num

    def get_bins_num(self):

        return self.__bins_num

    def get_bin_width(self):

        return self.__bin_width

    def get_seed(self):

        return self.__seed

    def get_verbose(self):

        return self.__verbose

    def get_device(self):

        return self.__device

    def get_last_time(self):

        return self.__last_time

    def get_init_time(self):

        return self.__init_time

    def get_prior_sigma(self):

        return self.__prior_sigma

    def get_prior_B_ls(self):

        return self.__prior_B_ls

    def get_prior_B_x0_c(self):

        return self.__prior_B_x0_c

    def get_prior_C_Q(self):

        return self.__prior_C_Q

    def get_prior_lambda(self):

        return self.__prior_lambda

    def get_prior_k(self):

        return self.__prior_C_Q.shape[1]

    def get_bins_bounds(self):

        bounds = self.__init_time + torch.cat((
            torch.as_tensor([self.__init_time], device=self.__device),
            torch.cumsum(torch.ones(size=(self.__bins_num, ), device=self.__device) * self.__bin_width, dim=0)
        ))
        return bounds

    def get_xt(self, events_times_list: torch.Tensor, x0: torch.Tensor = None, v: torch.Tensor = None) -> torch.Tensor:

        if x0 is None or v is None:
            raise ValueError("x0 and v cannot be None!")

        assert len(events_times_list) == x0.shape[0] and x0.shape[0] == v.shape[1], \
            print( len(events_times_list), x0.shape, v.shape)

        # Compute the event indices and residual times
        events_bin_indices = utils.div(events_times_list, self.__bin_width)
        residual_time = utils.remainder(events_times_list, self.__bin_width)
        events_bin_indices[events_bin_indices == self.__bins_num] = self.__bins_num - 1

        # Compute the total displacements for each time-intervals and append a zero column to get rid of indexing issues
        cum_displacement = torch.cat((
            torch.zeros(1, x0.shape[0], self.__dim, device=self.__device),
            torch.cumsum(v * self.__bin_width, dim=0)
        )).view(-1, self.__dim)

        xt = x0 + torch.index_select(
            cum_displacement, dim=0, index=events_bin_indices * v.shape[1] + torch.arange(len(events_bin_indices))
        )
        # Finally, add the displacement on the interval that nodes lay on
        xt = xt + torch.mul(
            residual_time.unsqueeze(1),
            torch.index_select(
                v.view(-1, self.__dim), dim=0, index=events_bin_indices * v.shape[1] + torch.arange(len(events_bin_indices))
            )
        )

        return xt

    def get_pairwise_distances(self, times_list: torch.Tensor, node_pairs: torch.Tensor = None):

        if node_pairs is None:
            raise NotImplementedError("It should be implemented for every node pairs!")

        x_tilde, v_tilde = self.get_x0(), self.get_v()

        delta_x0 = torch.index_select(x_tilde, dim=0, index=node_pairs[0]) - \
                   torch.index_select(x_tilde, dim=0, index=node_pairs[1])
        delta_v = torch.index_select(v_tilde, dim=1, index=node_pairs[0]) - \
                  torch.index_select(v_tilde, dim=1, index=node_pairs[1])

        # delta_xt is a tensor of size len(times_list) x dim
        delta_xt = self.get_xt(events_times_list=times_list, x0=delta_x0, v=delta_v)
        # Compute the squared Euclidean distance
        norm = torch.norm(delta_xt, p=2, dim=1, keepdim=False) ** 2

        return norm

    def get_log_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor):
        # Get pairwise distances
        intensities = -self.get_pairwise_distances(times_list=times_list, node_pairs=node_pairs)
        # Add an additional axis for beta parameters for time dimension
        intensities += torch.index_select(self.get_beta(), dim=0, index=node_pairs[0]) + \
                       torch.index_select(self.get_beta(), dim=0, index=node_pairs[1])

        return intensities

    def get_intensity(self, times_list: torch.tensor, node_pairs: torch.tensor):

        return torch.exp(self.get_log_intensity(times_list, node_pairs))

    def get_log_intensity_sum(self, delta_x0: torch.Tensor, delta_v: torch.Tensor, beta_ij: torch.Tensor,
                              events_count: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):

        bin_bounds = self.get_bins_bounds()

        # delta_xt is a tensor of size (bins_num x node_pairs) x dim
        delta_xt = self.get_xt(
            events_times_list=torch.repeat_interleave(bin_bounds[:-1], delta_x0.shape[0], dim=0),
            x0=delta_x0.repeat(self.__bins_num, 1),
            v=delta_v.repeat(1, self.__bins_num, 1)
        ).reshape(self.__bins_num, delta_x0.shape[0], self.__dim)

        delta_xt_sq_norm = torch.norm(delta_xt, p=2, dim=2, keepdim=False) ** 2
        delta_vt_sq_norm = torch.norm(delta_v, p=2, dim=2, keepdim=False) ** 2
        delta_xtvt = torch.sum(delta_xt * delta_v, dim=2, keepdim=False)

        intensities = torch.sum(beta_ij * events_count.sum(dim=1, keepdim=False))
        intensities += -torch.sum(delta_xt_sq_norm * events_count.T)
        intensities += -2*torch.sum(delta_xtvt * alpha1.T)
        intensities += -torch.sum(delta_vt_sq_norm * alpha2.T)

        return intensities

    def get_intensity_integral(self, delta_x0: torch.Tensor, delta_v: torch.Tensor, beta_ij: torch.Tensor,
                               bin_bounds: torch.Tensor = None, sum_required: bool = True):

        if bin_bounds is None:
            bin_bounds = self.get_bins_bounds()

        delta_xt = self.get_xt(
            events_times_list=torch.cat([bin_bounds[:-1]] * delta_x0.shape[0]),
            x0=torch.repeat_interleave(delta_x0, repeats=len(bin_bounds)-1, dim=0),
            v=torch.repeat_interleave(delta_v, repeats=len(bin_bounds)-1, dim=1),
        ).reshape((delta_x0.shape[0], len(bin_bounds)-1,  self.get_dim())).transpose(0, 1)

        norm_delta_xt = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
        norm_delta_v = torch.norm(delta_v, p=2, dim=2, keepdim=False) + utils.EPS

        inv_norm_delta_v = 1.0 / (norm_delta_v)
        delta_xt_v = (delta_xt * delta_v).sum(dim=2, keepdim=False)
        r = delta_xt_v * inv_norm_delta_v

        term0 = 0.5 * torch.sqrt(torch.as_tensor(utils.PI, device=self.__device)) * inv_norm_delta_v
        term1 = torch.exp(beta_ij.unsqueeze(0) + (r ** 2) - (norm_delta_xt ** 2))

        upper_bounds = bin_bounds[1:] - self.__bin_width * torch.arange(len(bin_bounds)-1)
        lower_bounds = bin_bounds[:-1] - self.__bin_width * torch.arange(len(bin_bounds)-1)
        term2_u = torch.erf(upper_bounds.unsqueeze(1) * norm_delta_v + r)
        term2_l = torch.erf(lower_bounds.unsqueeze(1) * norm_delta_v + r)

        if not sum_required:
            return term0 * term1 * (term2_u - term2_l)

        return (term0 * term1 * (term2_u - term2_l)).sum(dim=0)

    def get_intensity_integral_for(self, i: int, j: int, interval: torch.Tensor = None, distance: str = "squared_euc"):

        if distance == "riemann":
            sample_time_list = torch.linspace(interval[0], interval[1], steps=10000)
            delta_t = sample_time_list[1] - sample_time_list[0]
            riemann_integral_sum_lower_bound = 0
            for sample_t in sample_time_list[:-1]:
                riemann_integral_sum_lower_bound += self.get_intensity(
                    times_list=torch.as_tensor([sample_t]), node_pairs=torch.as_tensor([[i, j]]).T
                )
            riemann_integral_sum_lower_bound = riemann_integral_sum_lower_bound * delta_t

            return riemann_integral_sum_lower_bound

        # Expand the interval with bin bounds
        temp_interval = torch.arange(self.__bins_num+1, dtype=torch.float)*self.__bin_width
        mask = interval[0] <= temp_interval
        temp_interval = temp_interval[mask]
        mask = interval[1] >= temp_interval
        temp_interval = temp_interval[mask]

        if len(temp_interval):
            if interval[0] != temp_interval[0]:
                temp_interval = torch.cat((torch.as_tensor([interval[0]], dtype=torch.float), temp_interval))
            if interval[1] != temp_interval[-1]:
                temp_interval = torch.cat((temp_interval, torch.as_tensor([interval[1]], dtype=torch.float)))
        else:
            temp_interval = interval
        interval = temp_interval
        interval_idx = utils.div(interval, self.__bin_width)
        interval_idx[interval_idx == self.get_bins_num()] = self.get_bins_num() - 1

        x0, v, beta = self.get_x0(), self.get_v(), self.get_beta()

        unique_node_pairs = torch.as_tensor([i, j], dtype=torch.int, device=self.__device).t()

        # Common variables
        delta_x0 = torch.index_select(x0, dim=0, index=unique_node_pairs[0]) - \
                   torch.index_select(x0, dim=0, index=unique_node_pairs[1])
        delta_v = torch.index_select(v, dim=1, index=unique_node_pairs[0]) - \
                  torch.index_select(v, dim=1, index=unique_node_pairs[1])
        beta_ij = torch.index_select(beta, dim=0, index=unique_node_pairs[0]) + \
                  torch.index_select(beta, dim=0, index=unique_node_pairs[1])

        if distance != "squared_euc":
            raise ValueError("Invalid distance metric!")

        delta_xt = self.get_xt(
            events_times_list=torch.as_tensor(interval[:-1]),
            x0=torch.repeat_interleave(delta_x0, repeats=len(interval)-1, dim=0),
            v=torch.repeat_interleave(delta_v, repeats=len(interval)-1, dim=1),
        ).unsqueeze(1)

        delta_v = torch.index_select(delta_v, index=interval_idx[:-1], dim=0)

        norm_delta_xt = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
        # norm_v: a matrix of bins_counts x len(node_pairs)
        norm_delta_v = torch.norm(delta_v, p=2, dim=2, keepdim=False) + utils.EPS
        inv_norm_delta_v = 1.0 / (norm_delta_v)
        delta_xt_v = (delta_xt * delta_v).sum(dim=2, keepdim=False)
        r = delta_xt_v * inv_norm_delta_v

        term0 = 0.5 * torch.sqrt(torch.as_tensor(utils.PI, device=self.__device)) * inv_norm_delta_v
        term1 = torch.exp(beta_ij.unsqueeze(0) + (r ** 2) - (norm_delta_xt ** 2))

        upper_bounds = interval[1:].clone()
        upper_width_idx = utils.remainder(upper_bounds, self.__bin_width) < utils.EPS
        upper_bounds = utils.remainder(upper_bounds, self.__bin_width)
        upper_bounds[upper_width_idx] = self.__bin_width

        lower_bounds = interval[:-1].clone()
        lower_bounds = utils.remainder(lower_bounds, self.__bin_width)

        term2_u = torch.erf(upper_bounds.unsqueeze(
            1) * norm_delta_v + r)
        term2_l = torch.erf(lower_bounds.unsqueeze(
            1) * norm_delta_v + r)

        return (term0 * term1 * (term2_u - term2_l)).sum()

    def get_survival_function(self, times_list: torch.Tensor, x0: torch.Tensor = None, v: torch.Tensor = None,
                              beta: torch.Tensor = None, bin_bounds: torch.Tensor = None,
                              node_pairs: torch.tensor = None, distance: str = "squared_euc"):

        raise NotImplementedError("Survival Function Not Implemented!")
        # if x0 is None or v is None or bin_bounds is None:
        #     x0, v = self.get_x0(), self.get_v()
        #     bin_bounds = self.get_bins_bounds()
        #
        # if beta is None:
        #     beta = self.get_beta()
        #
        # if node_pairs is None:
        #     node_pairs = torch.triu_indices(self.__nodes_num, self.__nodes_num, offset=1)
        #
        # # Common variables
        # delta_x0 = x0[node_pairs[0], :] - x0[node_pairs[1], :]
        # delta_v = v[:, node_pairs[0], :] - v[:, node_pairs[1], :]
        # beta_ij = (beta**2).expand(len(node_pairs[0]))
        #
        # all_indices = torch.as_tensor([0] * len(bin_bounds) + [1] * len(times_list), dtype=torch.int)
        # all_bounds = torch.hstack((bin_bounds, times_list))
        # all_bounds, sorting_idx = torch.sort(all_bounds)
        # sorted_indices = all_indices[sorting_idx]
        # all_indices = torch.cumsum(sorted_indices, dim=0)
        # all_indices[all_indices % 2 == 0] = 0
        # # all_indices[sorted_indices] = 1
        # # all_indices = 1 - all_indices
        # all_indices = (all_indices == 0).nonzero()
        #
        # # Time indices
        # time_indices = torch.bucketize(all_bounds[:-1], boundaries=bin_bounds[1:-1], right=True)
        #
        # if distance == "squared_euc":
        #
        #     delta_xt = self.get_xt(times_list=all_bounds[:-1], x0=delta_x0, v=delta_v, bin_bounds=bin_bounds)
        #
        #     norm_delta_xt = torch.norm(delta_xt, p=2, dim=2, keepdim=False)
        #     # norm_v: a matrix of bins_counts x len(node_pairs)
        #     norm_delta_v = torch.norm(delta_v.index_select(0, time_indices), p=2, dim=2, keepdim=False) + utils.EPS
        #     inv_norm_delta_v = 1.0 / (norm_delta_v)
        #     # print(delta_xt.shape, delta_v[time_indices, :, :].shape)
        #     delta_xt_v = (delta_xt * delta_v.index_select(0, time_indices)).sum(dim=2, keepdim=False)
        #     r = delta_xt_v * inv_norm_delta_v
        #
        #     term0 = 0.5 * torch.sqrt(utils.PI).to(self.get_device()) * inv_norm_delta_v
        #     # term1 = torch.exp( beta_ij.unsqueeze(0) + r**2 - norm_delta_xt**2 )
        #     term1 = torch.exp(beta_ij.unsqueeze(0) + r ** 2 - norm_delta_xt ** 2)
        #     term2_u = torch.erf(all_bounds[1:].expand(norm_delta_v.shape[1], len(all_bounds)-1).t()*norm_delta_v + r)
        #     term2_l = torch.erf(all_bounds[:-1].expand(norm_delta_v.shape[1], len(all_bounds)-1).t()*norm_delta_v + r)
        #
        #     #
        #     diff = term2_u - term2_l
        #     diff = diff[all_indices[:-1]]
        #
        #     return (term0 * term1 * diff).sum(dim=0)
        #
        # else:
        #
        #     raise ValueError("Invalid distance metric!")

    def get_negative_log_likelihood(self, all_pairs: torch.Tensor,
                                    events_count: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor):

        x0_, v_, beta_ = self.get_x0(), self.get_v(), self.get_beta()
        delta_x0 = torch.index_select(x0_, dim=0, index=all_pairs[0]) - \
                  torch.index_select(x0_, dim=0, index=all_pairs[1])
        delta_v = torch.index_select(v_, dim=1, index=all_pairs[0]) - \
                  torch.index_select(v_, dim=1, index=all_pairs[1])
        beta_ij = torch.index_select(beta_, 0, all_pairs[0]) + \
                  torch.index_select(beta_, 0, all_pairs[1])

        non_integral_term = self.get_log_intensity_sum(
            delta_x0=delta_x0, delta_v=delta_v, beta_ij=beta_ij,
            events_count=events_count, alpha1=alpha1, alpha2=alpha2
        )
        integral_term = -self.get_intensity_integral(
            delta_x0=delta_x0, delta_v=delta_v, beta_ij=beta_ij
        ).sum()

        return -(non_integral_term + integral_term)

    def get_survival_log_likelihood(self, time_seq_list: list, node_pairs: torch.tensor):

        raise NotImplementedError("Survival not implemented!")
        # integral_term = 0
        # non_integral_term = 0
        # for idx in range(node_pairs.shape[1]):
        #
        #     times_list = time_seq_list[idx]
        #     node_pair = node_pairs[:, idx]
        #
        #     # if len(times_list):
        #
        #     integral_term += -self.get_survival_function(
        #         times_list=torch.as_tensor(times_list, device=self.__device), node_pairs=torch.as_tensor(node_pair).unsqueeze(1)
        #     ).sum()
        #
        #     non_integral_term += torch.sum(
        #         self.get_log_intensity(times_list=torch.as_tensor(times_list, device=self.__device),
        #                                node_pairs=torch.as_tensor(node_pair).unsqueeze(1)))
        #
        # return -(integral_term + non_integral_term)

    @staticmethod
    def get_B_factor(bin_centers1: torch.Tensor, bin_centers2: torch.Tensor,
                     prior_B_x0_c: torch.Tensor, prior_B_ls: torch.Tensor, only_kernel=False):

        prior_B_x0_c_sq = prior_B_x0_c ** 2
        time_mat = bin_centers1 - bin_centers2.T
        # prior_B_ls = torch.clamp(prior_B_ls, min=-1./bin_centers1.shape[1], max=1./bin_centers1.shape[1])
        # prior_B_ls = torch.clamp(prior_B_ls, min=-1e+2, max=1e+2)
        B_ls_sq = prior_B_ls ** 2
        kernel = torch.exp(-torch.div(time_mat ** 2, B_ls_sq)/2.0)

        # Combine the entry required for x0 with the velocity vectors covariance
        kernel = torch.block_diag(prior_B_x0_c_sq, kernel)

        # Add a constant term to get rid of computational problems and singularity
        kernel = kernel + (10*utils.EPS) * torch.eye(n=kernel.shape[0], m=kernel.shape[1])

        if only_kernel:
            return kernel

        # B x B lower triangular matrix
        L = torch.linalg.cholesky(kernel)

        return L

    @staticmethod
    def get_C_factor(prior_C_Q):
        # N x K matrix
        return torch.softmax(prior_C_Q, dim=1)

    @staticmethod
    def get_D_factor(dim):
        # D x D matrix
        return torch.eye(dim)

    def get_neg_log_prior(self, batch_nodes, batch_num=0):

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

        # B x B matrix
        B_factor = self.get_B_factor(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds,
            prior_B_x0_c=self.get_prior_B_x0_c(), prior_B_ls=self.get_prior_B_ls()
        )
        # N x K matrix where K is the community size
        C_factor = self.get_C_factor(prior_C_Q=self.get_prior_C_Q())
        # D x D matrix
        D_factor = self.get_D_factor(dim=self.get_dim())

        # B(batch_size)D x BKD matrix
        K_factor_batch = torch.kron(
            B_factor.contiguous(),
            torch.kron(torch.index_select(C_factor.contiguous(), dim=0, index=batch_nodes), D_factor).contiguous()
        )

        # Some common parameters
        lambda_sq = self.get_prior_lambda() ** 2
        sigma_sq = torch.clamp(self.get_prior_sigma(), min=5. / self.get_bins_num()) ** 2
        sigma_sq_inv = 1.0 / sigma_sq
        final_dim = self.get_number_of_nodes() * (self.get_bins_num()+1) * self.get_dim()
        reduced_dim = self.get_prior_k() * (self.get_bins_num()+1) * self.get_dim()

        # Compute the capacitance matrix R only if batch_num == 0
        if batch_num == 0:
            #K_factor_full = torch.kron(B_factor.contiguous(), torch.kron(C_factor.contiguous(), D_factor).contiguous())
            #self.__R = torch.eye(reduced_dim) + sigma_sq_inv * K_factor_full.T @ K_factor_full
            self.__R = torch.eye(reduced_dim) + sigma_sq_inv * torch.kron(
                B_factor.T @ B_factor, torch.kron(C_factor.T @ C_factor, D_factor.T @ D_factor)
            )
            self.__R_factor = torch.linalg.cholesky(self.__R)
            self.__R_factor_inv = torch.inverse(self.__R_factor)

        # Normalize and vectorize the initial position and velocity vectors
        x0_batch = torch.index_select(self.get_x0(), dim=0, index=batch_nodes).flatten()
        v_batch = utils.vectorize(torch.index_select(self.get_v(),  dim=1, index=batch_nodes)).flatten()
        x0v = torch.hstack((x0_batch, v_batch))

        # Computation of the squared Mahalanobis distance: v.T @ inv(D + W @ W.T) @ v
        # It uses Woodbury matrix identity: inv(D + Kf @ Kf.T) = inv(D) - inv(D) @ Kf @ inv(R) @ Kf.T @ inv(D),
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        mahalanobis_term1 = sigma_sq_inv * x0v.pow(2).sum(-1)
        mahalanobis_term2 = (sigma_sq_inv * x0v @ K_factor_batch @ self.__R_factor_inv.T).pow(2).sum(-1)
        m = (1.0 / lambda_sq) * (mahalanobis_term1 - mahalanobis_term2 )

        # Computation of the log determinant
        # It uses Matrix Determinant Lemma: log|D + Kf @ Kf.T| = log|R| + log|D|,
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        log_det = 2 * self.__R_factor.diagonal(dim1=-2, dim2=-1).log().sum(-1) + final_dim * (lambda_sq.log() + sigma_sq.log())

        # Compute the negative log-likelihood
        log_prior_likelihood = -0.5 * (final_dim * utils.LOG2PI + log_det + m)

        return -log_prior_likelihood.squeeze(0)


    def average_velocity(self):

        return torch.mean(self.__v, dim=0)