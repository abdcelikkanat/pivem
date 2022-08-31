from src.learning import LearningModel
from src.base import BaseModel
from utils import *


class PredictionModel(BaseModel, torch.nn.Module):

    def __init__(self, x0: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, bins_num: int = 100,
                 last_time: float = 1.0, prior_lambda: float = 1e5, prior_sigma: torch.Tensor = None,
                 prior_B_x0_c: torch.Tensor = None, prior_B_sigma: torch.Tensor = None, prior_C_Q: torch.Tensor= None,
                 device: torch.device = "cpu", verbose: bool = False, seed: int = 0, **kwargs):

        kwargs = kwargs

        # super(LearningModel, self).__init__(
        #     x0=kwargs["x0"],
        #     v=torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False),
        #     beta=torch.nn.Parameter(2 * torch.rand(size=(nodes_num,), device=device) - 1, requires_grad=False),
        #     bins_num=bins_num,
        #     last_time=last_time,
        #     prior_lambda=prior_lambda,
        #     prior_sigma=torch.nn.Parameter(
        #         (2.0 / bins_num) * torch.rand(size=(1,), device=device) + (1. / bins_num), requires_grad=False
        #     ),
        #     prior_B_x0_c=torch.nn.Parameter(torch.ones(size=(1, 1), device=device), requires_grad=False),
        #     prior_B_sigma=torch.nn.Parameter(
        #         (1 - (2.0 / bins_num)) * torch.rand(size=(1,), device=device) + (1. / bins_num), requires_grad=False
        #     ),
        #     prior_C_Q=torch.nn.Parameter(torch.rand(size=(nodes_num, prior_k), device=device), requires_grad=False),
        #     device=device,
        #     verbose=verbose,
        #     seed=seed
        # )

        # super(PredictionModel, self).__init__(
        #     **{key.replace('_BaseModel__', ''): value for key, value in kwargs.items()}
        # )
        super(PredictionModel, self).__init__(
            x0=x0, v=v, beta=beta, bins_num=bins_num, last_time=last_time, prior_lambda=prior_lambda,
            prior_sigma=prior_sigma, prior_B_x0_c=prior_B_x0_c, prior_B_sigma=prior_B_sigma, prior_C_Q=prior_C_Q,
            device=device, verbose=verbose, seed=seed, kwargs=kwargs
        )

        self.__inv_train_cov = self.__compute_inv_train_cov()

    def get_inv_train_cov(self):

        return self.__inv_train_cov

    def __compute_inv_train_cov(self):

        # Some scalars
        sigma_sq = torch.clamp(self.get_prior_sigma(), min=5. / self.get_bins_num()) ** 2
        sigma_sq_inv = 1.0 / sigma_sq
        lambda_sq = self.get_prior_lambda() ** 2
        reduced_dim = self.get_prior_k() * (self.get_bins_num() + 1) * self.get_dim()
        final_dim = self.get_number_of_nodes() * (self.get_bins_num() + 1) * self.get_dim()

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

        # B x B matrix
        B_factor = self.get_B_factor(
            bin_centers1=middle_bounds, bin_centers2=middle_bounds,
            prior_B_x0_c=self.get_prior_B_x0_c(), prior_B_sigma=self.get_prior_B_sigma()
        )
        # N x K matrix where K is the community size
        C_factor = self.get_C_factor(prior_C_Q=self.get_prior_C_Q())
        # D x D matrix
        D_factor = self.get_D_factor(dim=self.get_dim())

        # Compute the capacitance matrix R only if batch_num == 0
        R = torch.eye(reduced_dim) + sigma_sq_inv * torch.kron(
            B_factor.T @ B_factor, torch.kron(C_factor.T @ C_factor, D_factor.T @ D_factor)
        )
        R_factor = torch.linalg.cholesky(R)
        R_factor_inv = torch.inverse(R_factor)

        # Computation of inv(D + W @ W.T)
        # It uses Woodbury matrix identity: inv(D + Kf @ Kf.T) = inv(D) - inv(D) @ Kf @ inv(R) @ Kf.T @ inv(D),
        # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
        term1 = torch.diag(sigma_sq_inv*torch.ones(size=(final_dim, ), dtype=torch.float, device=self.get_device()))
        K_factor = torch.kron(B_factor.contiguous(), torch.kron(C_factor, D_factor))
        f = (sigma_sq_inv * K_factor @ R_factor_inv.T)
        term2 = f @ f.T

        return (1.0 / lambda_sq) * (term1 - term2)

    def get_v_pred(self, times_list: torch.Tensor):

        # Get the bin bounds
        bounds = self.get_bins_bounds()

        # Get the middle time points of the bins for TxT covariance matrix
        middle_bounds = ((bounds[1:] + bounds[:-1]) / 2.).view(1, self.get_bins_num())

        # (B+1) x len(times_list) matrix
        B = self.get_B_factor(
            bin_centers1=times_list.view(1, len(times_list)), bin_centers2=middle_bounds,
            prior_B_x0_c=self.get_prior_B_x0_c(), prior_B_sigma=self.get_prior_B_sigma(), only_kernel=True
        )
        B = B[:, 1:]  # remove the first row corresponding to initial position vectors

        # N x K matrix where K is the community size
        C_factor = self.get_C_factor(prior_C_Q=self.get_prior_C_Q())
        # D x D matrix
        D_factor = self.get_D_factor(dim=self.get_dim())

        # Construct the test_train covariance matrix
        test_train_cov = torch.kron(B, torch.kron((C_factor @ C_factor.T), (D_factor @ D_factor.T)))

        # Normalize and vectorize the initial position and velocity vectors
        x0 = vectorize(self.get_x0())
        v_batch = vectorize(self.get_v()).flatten()

        # Compute the estimated velocity vectors
        x0v = torch.hstack((x0, v_batch))
        est = unvectorize(
            test_train_cov.T @ self.get_inv_train_cov() @ x0v,
            size=(len(times_list), self.get_number_of_nodes(), self.get_dim())
        )

        return est

    def get_x_pred(self, times_list: torch.Tensor):

        # A matrix of size N x D
        x_last = self.get_xt(
            events_times_list=torch.as_tensor([self.get_last_time()]*self.get_number_of_nodes()),
            x0=self.get_x0(), v=self.get_v()
        )

        # Get the estimated velocity matrix of size len(time_samples) x N x D
        pred_v = self.get_v_pred(times_list=times_list)

        # A matrix of size len(time_samples) x N x D
        pred_x = x_last.unsqueeze(0) + (times_list - self.get_last_time()).view(-1, 1, 1) * pred_v

        return pred_x

    def get_intensity_integral_pred(self, t_init: float, t_last: float, sample_size=100):

        assert t_init >= self.get_last_time(), \
            "The given boundary times must be larger than the last time of the training data!"

        time_samples = torch.linspace(t_init, t_last, steps=sample_size)[:-1]  # Discard the last time point
        delta_t = time_samples[1] - time_samples[0]

        # N x N
        beta_mat = self.get_beta().unsqueeze(1) + self.get_beta().unsqueeze(0)
        # (sample_size-1) x N x D
        xt = self.get_x_pred(times_list=time_samples)
        # (sample_size-1) x (N x N)
        delta_x_mat = torch.cdist(xt, xt, p=2)
        # (sample_size-1) x (N x N)
        lambda_pred = torch.sum(torch.exp(beta_mat.unsqueeze(0) - delta_x_mat), dim=0) * delta_t

        return lambda_pred


        # self.get_v().requires_grad = False
        #
        #
        # v_batch = mean_normalization(torch.index_select(self._v, dim=1, index=nodes))
        # v_vect_batch = utils.vectorize(v_batch).flatten()
        #
        # # Get the bin bounds
        # bounds = self._lm.get_bins_bounds()
        # # Get the middle time points of the bins for TxT covariance matrix
        # middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, len(bounds) - 1) / 2.
        #
        # # N x K matrix where K is the community size
        # C_factor = self._lm._get_C_factor().T
        # # D x D matrix
        # D_factor = self._lm._get_D_factor()
        #
        # B_test_train_factor = self._lm._get_B_factor(
        #     bin_centers1=middle_bounds, bin_centers2=times_list.view(1, 1, len(times_list)), only_kernel=True
        # )
        # test_train_cov = torch.kron(B_test_train_factor, torch.kron(C_factor @ C_factor.T, D_factor @ D_factor.T))
        #
        # mean_vt = test_train_cov @ self._train_cov_inv @ v_vect_batch
        #
        # return utils.unvectorize(mean_vt, size=(len(times_list), v_batch.shape[1], v_batch.shape[2]))



        # super(BaseModel, self).__init__(
        #     x0=torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False),
        #     v=torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False),
        #     beta=torch.nn.Parameter(2 * torch.rand(size=(nodes_num,), device=device) - 1, requires_grad=False),
        #     bins_num=bins_num,
        #     last_time=last_time,
        #     prior_lambda=prior_lambda,
        #     prior_sigma=torch.nn.Parameter(
        #         (2.0 / bins_num) * torch.rand(size=(1,), device=device) + (1. / bins_num), requires_grad=False
        #     ),
        #     prior_B_x0_c=torch.nn.Parameter(torch.ones(size=(1, 1), device=device), requires_grad=False),
        #     prior_B_sigma=torch.nn.Parameter(
        #         (1 - (2.0 / bins_num)) * torch.rand(size=(1,), device=device) + (1. / bins_num), requires_grad=False
        #     ),
        #     prior_C_Q=torch.nn.Parameter(torch.rand(size=(nodes_num, prior_k), device=device), requires_grad=False),
        #     device=device,
        #     verbose=verbose,
        #     seed=seed
        # )


    #
    #
    #
    #     super(PredictionModel, self).__init__()
    #
    #     # Set the learning model
    #     self._lm = lm
    #     # Set the parameters of Learning model
    #     for key, value in self._lm.get_hyperparameters().items():
    #         setattr(self, key, value)
    #     # Set the initial and last time points
    #     self._pred_init_time = pred_init_time
    #     self._pred_last_time = pred_last_time
    #
    #     # Sample some time points
    #     self._num_of_samples = num_of_samples
    #     self._time_samples = torch.linspace(self._pred_init_time, self._pred_last_time, self._num_of_samples)
    #     self._time_delta = self._time_samples[1] - self._time_samples[0]
    #
    #     # A tensor of len(time_samples) x _nodes_num x dim
    #     self._train_cov_inv = self.get_train_cov_inv()
    #     self._x_init = self._lm.get_xt(
    #         events_times_list=torch.as_tensor([self._pred_init_time]*self._nodes_num), x0=self._x0, v=self._v
    #     )
    #
    #     self._time_samples_expected_v = self.get_expected_vt(times_list=self._time_samples)
    #
    # def get_train_cov_inv(self,  nodes: torch.Tensor = None):
    #
    #     if nodes is not None:
    #         raise ValueError("It has been implemented for the whole node set!")
    #
    #     # Get the bin bounds
    #     bounds = self._lm.get_bins_bounds()
    #     # Get the middle time points of the bins for TxT covariance matrix
    #     middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, len(bounds)-1) / 2.
    #
    #     # B x B matrix
    #     B_factor = self._lm._get_B_factor(bin_centers1=middle_bounds, bin_centers2=middle_bounds)
    #     # N x K matrix where K is the community size
    #     C_factor = self._lm._get_C_factor().T
    #     # D x D matrix
    #     D_factor = self._lm._get_D_factor()
    #
    #     # Some common parameters
    #     lambda_sq = self._prior_lambda ** 2
    #     sigma_sq = torch.sigmoid(self._prior_sigma)
    #     sigma_sq_inv = 1.0 / sigma_sq
    #     final_dim = self._lm.get_number_of_nodes() * (len(bounds)-1) * self._dim
    #     reduced_dim = self._prior_C_Q.shape[0] * (len(bounds)-1) * self._dim
    #
    #     K_factor = torch.kron(B_factor.contiguous(), torch.kron(C_factor.contiguous(), D_factor).contiguous())
    #
    #     R = torch.eye(reduced_dim) + sigma_sq_inv * K_factor.T @ K_factor
    #     R_inv = torch.cholesky_inverse(torch.linalg.cholesky(R))
    #
    #     # Compute the inverse of covariance matrix
    #     # Computation of the squared Mahalanobis distance: v.T @ inv(D + W @ W.T) @ v
    #     # It uses Woodbury matrix identity: inv(D + Kf @ Kf.T) = inv(D) - inv(D) @ Kf @ inv(R) @ Kf.T @ inv(D),
    #     # where R is the capacitance matrix defined by I + Kf.T @ inv(D) @ Kf
    #     mahalanobis_term1 = sigma_sq_inv * torch.eye(final_dim)
    #     mahalanobis_term2 = sigma_sq_inv * K_factor @ R_inv @ K_factor.T * sigma_sq_inv
    #     train_cov_inv = (1.0 / lambda_sq) * (mahalanobis_term1 - mahalanobis_term2)
    #
    #     return train_cov_inv
    #
    # def get_expected_vt(self, times_list: torch.Tensor):
    #
    #     nodes = torch.arange(self._nodes_num)
    #
    #     # Normalize and vectorize the velocities
    #     v_batch = mean_normalization(torch.index_select(self._v, dim=1, index=nodes))
    #     v_vect_batch = utils.vectorize(v_batch).flatten()
    #
    #     # Get the bin bounds
    #     bounds = self._lm.get_bins_bounds()
    #     # Get the middle time points of the bins for TxT covariance matrix
    #     middle_bounds = (bounds[1:] + bounds[:-1]).view(1, 1, len(bounds) - 1) / 2.
    #
    #     # N x K matrix where K is the community size
    #     C_factor = self._lm._get_C_factor().T
    #     # D x D matrix
    #     D_factor = self._lm._get_D_factor()
    #
    #     B_test_train_factor = self._lm._get_B_factor(
    #         bin_centers1=middle_bounds, bin_centers2=times_list.view(1, 1, len(times_list)), only_kernel=True
    #     )
    #     test_train_cov = torch.kron(B_test_train_factor, torch.kron(C_factor @ C_factor.T, D_factor @ D_factor.T))
    #
    #     mean_vt = test_train_cov @ self._train_cov_inv @ v_vect_batch
    #
    #     return utils.unvectorize(mean_vt, size=(len(times_list), v_batch.shape[1], v_batch.shape[2]))
    #
    # def get_expected_displacements(self, times_list: torch.Tensor, nodes: torch.Tensor):
    #
    #     expected_vt = torch.index_select(self.get_expected_vt(times_list=times_list), dim=1, index=nodes)
    #
    #     events_bin_indices = torch.div(
    #         times_list - self._pred_init_time, self._time_delta, rounding_mode='floor'
    #     ).type(torch.int)
    #     residual_time = (times_list - self._pred_init_time) % self._time_delta
    #     events_bin_indices[events_bin_indices == len(self._time_samples) - 1] = len(self._time_samples) - 2
    #
    #     # Riemann integral for computing average displacement
    #     xt_disp = torch.cumsum(
    #         self._time_delta * torch.index_select(self._time_samples_expected_v, dim=1, index=nodes), dim=0
    #     )
    #     xt_disp = torch.index_select(xt_disp, dim=0, index=events_bin_indices)
    #
    #     # Remaining displacement
    #     remain_disp = torch.mul(expected_vt, residual_time.unsqueeze(1))
    #
    #     # Get average position
    #     mean_xt = torch.index_select(self._x_init, dim=0, index=nodes).unsqueeze(0) + xt_disp + remain_disp
    #
    #     return mean_xt
    #
    # def get_log_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor):
    #
    #     # Sum of bias terms
    #     intensities = torch.index_select(self._beta, dim=0, index=node_pairs[0]) + \
    #                   torch.index_select(self._beta, dim=0, index=node_pairs[1])
    #
    #     for idx in range(node_pairs.shape[1]):
    #
    #         pair_xt = self.get_expected_displacements(
    #             times_list=times_list[idx].unsqueeze(0), nodes=node_pairs[:, idx]
    #         ).squeeze(0)
    #         pair_norm = torch.norm(pair_xt[0, :] - pair_xt[1, :], p=2, dim=0, keepdim=False) ** 2
    #
    #         intensities[idx] = intensities[idx] - pair_norm
    #
    #     return intensities
    #
    # def get_intensity(self, times_list: torch.Tensor, node_pairs: torch.Tensor):
    #
    #     return torch.exp(self.get_log_intensity(times_list=times_list, node_pairs=node_pairs))
    #
    # def get_intensity_integral(self, nodes: torch.Tensor):
    #
    #     return self._lm.get_intensity_integral(
    #         nodes=nodes, x0=self._x_init, v=self._time_samples_expected_v[:-1, :, :],
    #         beta=self._beta, bin_bounds=self._time_samples
    #     )
    #
    # def get_intensity_integral_for_bins(self, boundaries: torch.Tensor, sample_size_per_bin=100):
    #
    #     assert boundaries[0] == self._pred_init_time and boundaries[-1] == self._pred_last_time, \
    #         "The first and the last time points are wrong!"
    #
    #     # If the sample_size_per_bin is integer, then sample an equal number of points from the timeline
    #     if type(sample_size_per_bin) is int:
    #         sample_size_per_bin = sample_size_per_bin * torch.ones(size=(len(boundaries)-1, ), dtype=torch.int)
    #
    #     # Sample 'sample_size_per_bin[i]' number of points for each interval
    #     chosen_time_points = torch.as_tensor([boundaries[0]])
    #     for b in range(len(boundaries)-1):
    #         bin_samples = torch.linspace(start=boundaries[b], end=boundaries[b+1], steps=sample_size_per_bin[b]+1)
    #         chosen_time_points = torch.cat((chosen_time_points, bin_samples[1:]))
    #
    #     expected_v = self.get_expected_vt(times_list=chosen_time_points)
    #     nodes = torch.arange(self._nodes_num)
    #
    #     # len(sum(sample_size_per_bin)) x (self._nodes_num x (self._nodes_num-1)/2) matrix
    #     intensities = self._lm.get_intensity_integral(
    #         nodes=nodes, x0=self._x_init, v=expected_v[:-1, :, :],
    #         beta=self._beta, bin_bounds=chosen_time_points, sum=False
    #     )
    #
    #     # Sum the intensities integrals by choosing correct indices
    #     index_cumsum = torch.cumsum(sample_size_per_bin, dim=0) - 1
    #     intensities_cumsum = torch.cumsum(intensities, dim=0)
    #     integrals = torch.index_select(intensities_cumsum, index=index_cumsum, dim=0)
    #
    #     # A matrix of size ( len(boundaries)-1) x (self._nodes_num x (self._nodes_num-1)/2) )
    #     return integrals
    #
    # def get_negative_log_likelihood(self, event_times: torch.Tensor, event_node_pairs: torch.Tensor):
    #
    #     nodes = torch.arange(self._nodes_num)
    #
    #     integral_term_all_pairs = -self._lm.get_intensity_integral(
    #         nodes=nodes, x0=self._x_init, v=self._time_samples_expected_v[:-1, :, :],
    #         beta=self._beta, bin_bounds=self._time_samples
    #     )
    #
    #     integral_term = torch.as_tensor(
    #         [integral_term_all_pairs[utils.pairIdx2flatIdx(p[0], p[1], self._nodes_num)] for p in event_node_pairs.T],
    #         dtype=torch.float
    #     )
    #
    #     non_integral_term = self.get_log_intensity(times_list=event_times, node_pairs=event_node_pairs)
    #
    #     return -(non_integral_term + integral_term)



