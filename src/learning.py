import sys
import math
import torch
from src.base import BaseModel
import time
import utils


class LearningModel(BaseModel, torch.nn.Module):

    def __init__(self, data, nodes_num, bins_num, dim, last_time: float, approach: str = "nhpp",
                 prior_k: int = 10, prior_lambda: float = 1.0, masked_pairs: torch.Tensor = None,
                 learning_rate: float = 0.1, batch_size: int = None, epoch_num: int = 100,
                 steps_per_epoch=10, device: torch.device = None, verbose: bool = False, seed: int = 19):

        super(LearningModel, self).__init__(
            x0=torch.nn.Parameter(2. * torch.rand(size=(nodes_num, dim), device=device) - 1., requires_grad=False),
            v=torch.nn.Parameter(torch.zeros(size=(bins_num, nodes_num, dim), device=device), requires_grad=False),
            beta=torch.nn.Parameter(2 * torch.rand(size=(nodes_num, ), device=device) - 1, requires_grad=False),
            bins_num=bins_num,
            last_time=last_time,
            prior_lambda=prior_lambda,
            prior_sigma=torch.nn.Parameter(
                (2.0 / bins_num) * torch.rand(size=(1,), device=device) + (1./bins_num), requires_grad=False
            ),
            prior_B_x0_c=torch.nn.Parameter(torch.ones(size=(1, 1), device=device), requires_grad=False),
            prior_B_ls=torch.nn.Parameter(
                (1 - (2.0 / bins_num)) * torch.rand(size=(1,), device=device) + (1./bins_num), requires_grad=False
            ),
            prior_C_Q=torch.nn.Parameter(torch.rand(size=(nodes_num, prior_k), device=device), requires_grad=False),
            device=device,
            verbose=verbose,
            seed=seed
        )

        # Get the pairs and events
        self.__events_pairs = data[0]
        self.__events = data[1]

        # The approach used for learning the representations [ nhpp or survival method ]
        self.__approach = approach

        # Parameters for optimization
        self.__learning_procedure = "seq"
        self.__learning_rate = learning_rate
        self.__epoch_num = epoch_num
        self.__batch_size = nodes_num if batch_size is None else batch_size
        self.__steps_per_epoch = steps_per_epoch
        self.__learning_param_names = [["x0", "beta", ], ["v"], ["reg_params"]]  # Order matters for sequential learning
        self.__learning_param_epoch_weights = [1, 1, 1]
        self.__optimizer = None

        # Node pairs which will be discarded during the optimization
        self.__masked_pairs = masked_pairs

        # A list to store the training losses
        self.__loss = []

        # Pre-computation of some coefficients
        self.__events_count, self.__alpha1, self.__alpha2 = self.compute_coefficients(
            self.get_number_of_nodes(), self.__events_pairs, self.__events, self.get_bins_num()
        )

    def compute_coefficients(self, nodes_num, events_pairs, events, bins_num):

        if self.get_verbose():
            print(f"+ The pre-computation of the coefficients has started.")
            init_time = time.time()

        # Initialization
        events_count = {
            utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=nodes_num):
                torch.zeros(size=(bins_num,), dtype=torch.int, device=self.get_device()) for pair in events_pairs
        }
        alpha1 = {
            utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=nodes_num):
                torch.zeros(size=(bins_num,), dtype=torch.float, device=self.get_device()) for pair in events_pairs
        }
        alpha2 = {
            utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=nodes_num):
                torch.zeros(size=(bins_num,), dtype=torch.float, device=self.get_device()) for pair in events_pairs
        }

        for pairIdx, pair in enumerate(events_pairs):
            # Get the corresponding index
            dictIdx = utils.pairIdx2flatIdx(i=pair[0], j=pair[1], n=nodes_num)

            # Get the bin indices
            bin_idx = utils.div(
                torch.as_tensor(events[pairIdx], dtype=torch.float, device=self.get_device()), self.get_bin_width()
            )
            bin_idx[bin_idx == bins_num] = bins_num - 1
            events_count[dictIdx].index_add_(
                dim=0, index=bin_idx,
                source=torch.ones(len(bin_idx), dtype=torch.int, device=self.get_device())
            )
            alpha1[dictIdx].index_add_(
                dim=0, index=bin_idx,
                source=utils.remainder(
                    torch.as_tensor(events[pairIdx], dtype=torch.float, device=self.get_device()),
                    self.get_bin_width()
                )
            )
            alpha2[dictIdx].index_add_(
                dim=0, index=bin_idx,
                source=utils.remainder(
                    torch.as_tensor(events[pairIdx], dtype=torch.float, device=self.get_device()),
                    self.get_bin_width()) ** 2
            )

        if self.get_verbose():
            print("\t+ Completed in {:.2f} secs.".format(time.time() - init_time))

        return events_count, alpha1, alpha2

    def learn(self, learning_type=None, loss_file_path=None):

        learning_type = self.__learning_procedure if learning_type is None else learning_type

        # Initialize optimizer list
        self.__optimizer = []

        if learning_type == "seq":

            # For each parameter group, add an optimizer
            for param_group in self.__learning_param_names:

                # Set the gradients to True
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": True})

                # Add a new optimizer
                self.__optimizer.append(
                    torch.optim.Adam(self.parameters(), lr=self.__learning_rate)
                )
                # Set the gradients to False
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": False})

            # Run alternating minimization
            self.__sequential_learning()

            if loss_file_path is not None:
                with open(loss_file_path, 'w') as f:
                    for batch_losses in self.__loss:
                        f.write(f"{' '.join('{:.3f}'.format(loss) for loss in batch_losses)}\n")

        elif learning_type == "alt":

            # For each parameter group, add an optimizer
            for param_group in self.__learning_param_names:

                # Set the gradients to True
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": True})
                # Add a new optimizer
                self.__optimizer.append(
                    torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.__learning_rate)
                )
                # Set the gradients to False
                for param_name in param_group:
                    self.__set_gradients(**{f"{param_name}_grad": False})

            # Run alternating minimization
            self.__alternating_learning()

        else:

            raise NotImplementedError("A learning method other than alternation minimization is not implemented!")

    def __sequential_learning(self):

        if self.get_verbose():
            print("- Training started (Sequential Learning).")

        current_epoch = 0
        current_param_group_idx = 0
        group_epoch_counts = (self.__epoch_num * torch.cumsum(
            torch.as_tensor([0] + self.__learning_param_epoch_weights, device=self.get_device(), dtype=torch.float), dim=0
        ) / sum(self.__learning_param_epoch_weights)).type(torch.int)
        group_epoch_counts = group_epoch_counts[1:] - group_epoch_counts[:-1]

        while current_epoch < self.__epoch_num:

            # Set the gradients to True
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": True})

            # Repeat the optimization of the group parameters given weight times
            for _ in range(group_epoch_counts[current_param_group_idx]):
                self.__train_one_epoch(
                    epoch_num=current_epoch, optimizer=self.__optimizer[current_param_group_idx]
                )
                current_epoch += 1

            # Iterate the parameter group id
            current_param_group_idx += 1

    def __alternating_learning(self):

        current_epoch = 0
        current_param_group_idx = 0
        while current_epoch < self.__epoch_num:

            # Set the gradients to True
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": True})

            # Repeat the optimization of the group parameters given weight times
            for _ in range(self.__learning_param_epoch_weights[current_param_group_idx]):

                self.__train_one_epoch(
                    epoch_num=current_epoch, optimizer=self.__optimizer[current_param_group_idx]
                )
                current_epoch += 1

            # Set the gradients to False
            for param_name in self.__learning_param_names[current_param_group_idx]:
                self.__set_gradients(**{f"{param_name}_grad": False})

            # Iterate the parameter group id
            current_param_group_idx = (current_param_group_idx + 1) % len(self.__learning_param_epoch_weights)

    def __train_one_epoch(self, epoch_num, optimizer):

        init_time = time.time()

        total_batch_loss = 0
        self.__loss.append([])
        for batch_num in range(self.__steps_per_epoch):

            batch_loss = self.__train_one_batch(batch_num)

            self.__loss[-1].append(batch_loss)

            total_batch_loss += batch_loss

        # Set the gradients to 0
        optimizer.zero_grad()

        # Backward pass
        total_batch_loss.backward()

        # Perform a step
        optimizer.step()

        # Get the average epoch loss
        epoch_loss = total_batch_loss / float(self.__steps_per_epoch)

        if not math.isfinite(epoch_loss):
            print(f"- Epoch loss is {epoch_loss}, stopping training")
            sys.exit(1)

        if self.get_verbose() and (epoch_num % 10 == 0 or epoch_num == self.__epoch_num - 1):
            time_diff = time.time() - init_time
            print("\t+ Epoch = {} | Loss/train: {} | Elapsed time: {:.2f}".format(epoch_num, epoch_loss, time_diff))

    def __train_one_batch(self, batch_num):

        self.train()

        batch_nodes = torch.multinomial(
            torch.ones(self.get_number_of_nodes(), dtype=torch.float, device=self.get_device()),
            self.__batch_size, replacement=False
        )
        batch_nodes, _ = torch.sort(batch_nodes, dim=0)
        batch_pairs = torch.combinations(batch_nodes, r=2).T.type(torch.int)

        # Remove the masked pairs from the batch pairs
        self.__masked_pairs = torch.as_tensor([[0, 1], [1, 2], [1, 3]], dtype=torch.int).T

        if self.__masked_pairs is not None:
            dist = torch.cdist(
                batch_pairs.T.unsqueeze(0).type(torch.float), self.__masked_pairs.T.unsqueeze(0).type(torch.float)
            ).squeeze(0)
            unmatched_indices = dist.nonzero()[:, 0]
            batch_pairs = torch.index_select(batch_pairs, dim=1, index=unmatched_indices)

        # Forward pass
        average_batch_loss = self.forward(
            nodes=batch_nodes, pairs=batch_pairs,
            events_count=torch.as_tensor(
                [self.__events_count.get(
                    utils.pairIdx2flatIdx(pair[0], pair[1], self.get_number_of_nodes()),
                    torch.zeros(size=(self.get_bins_num(), ), dtype=torch.int, device=self.get_device())
                ).tolist() for pair in batch_pairs.T.tolist()],
                dtype=torch.int
            ),
            alpha1=torch.as_tensor(
                [self.__alpha1.get(
                    utils.pairIdx2flatIdx(pair[0], pair[1], self.get_number_of_nodes()),
                    torch.zeros(size=(self.get_bins_num(),), dtype=torch.float, device=self.get_device())
                ).tolist() for pair in batch_pairs.T.tolist()],
                dtype=torch.float
            ),
            alpha2=torch.as_tensor(
                [self.__alpha2.get(
                    utils.pairIdx2flatIdx(pair[0], pair[1], self.get_number_of_nodes()),
                    torch.zeros(size=(self.get_bins_num(),), dtype=torch.float, device=self.get_device())
                ).tolist() for pair in batch_pairs.T.tolist()],
                dtype=torch.float
            ),
            batch_num=batch_num
        )

        return average_batch_loss

    def forward(self, nodes: torch.Tensor, pairs: torch.Tensor,
                events_count: torch.Tensor, alpha1: torch.Tensor, alpha2: torch.Tensor, batch_num: int):

        nll = 0
        if self.__approach == "nhpp":
            nll = nll + self.get_negative_log_likelihood(pairs, events_count, alpha1, alpha2)

        elif self.__approach == "survival":
            pass #nll += self.get_survival_log_likelihood(nodes, event_times, event_node_pairs)

        else:
            raise ValueError("Invalid approach name!")

        # Add prior
        nll = nll + self.get_neg_log_prior(batch_nodes=nodes, batch_num=batch_num)

        return nll

    def __set_gradients(self, beta_grad=None, x0_grad=None, v_grad=None, reg_params_grad=None):

        if beta_grad is not None:
            self.get_beta().requires_grad = beta_grad

        if x0_grad is not None:
            self.get_x0(standardize=False).requires_grad = x0_grad

        if v_grad is not None:
            self.get_v(standardize=False).requires_grad = v_grad

        if reg_params_grad is not None:

            # Set the gradients of the prior function
            for name, param in self.named_parameters():
                if '_prior' in name:
                    param.requires_grad = reg_params_grad

    def save(self, path):

        if self.get_verbose():
            print(f"- Model file is saving.")
            print(f"\t+ Target path: {path}")

        kwargs = {
            'data': [self.__events_pairs, self.__events ],
            'nodes_num': self.get_number_of_nodes(), 'bins_num': self.get_bins_num(), 'dim': self.get_dim(),
            'last_time': self.get_last_time(), 'approach': self.__approach,
            'prior_k': self.get_prior_k(), 'prior_lambda': self.get_prior_lambda(), 'masked_pairs': self.__masked_pairs,
            'learning_rate': self.__learning_rate, 'batch_size': self.__batch_size, 'epoch_num': self.__epoch_num,
            'steps_per_epoch': self.__steps_per_epoch,
            'device': self.get_device(), 'verbose': self.get_verbose(), 'seed': self.get_seed(),
            # 'learning_procedure': self.__learning_procedure,
            # 'learning_param_names': self.__learning_param_names,
            # 'learning_param_epoch_weights': self.__learning_param_epoch_weights
        }

        torch.save([kwargs, self.state_dict()], path)

        if self.get_verbose():
            print(f"\t+ Completed.")


