import re
import torch


class Dataset:
    """
    A class to read a temporal network dataset
    Version 2.0
    """

    def __init__(self, nodes_num=0, edges: torch.LongTensor = None, edge_times: torch.Tensor = None,
                 edge_weights: torch.Tensor = None, directed: bool = None, signed: bool = None, verbose=False):

        self.__nodes_num = nodes_num
        self.__edges = edges
        self.__times = edge_times
        self.__weights = edge_weights
        self.__directed = directed
        self.__signed = signed
        self.__verbose = verbose

        # If the edge times are given, set the initial and last time
        if self.__times is not None:
            self.__init_time = self.__times.min()
            self.__last_time = self.__times.max()

        # Check if the given parameters are valid
        if self.__edges is not None:
            assert self.__edges.shape[0] == 2, \
                "The edges must be a matrix of shape (2xT)!"
            assert self.__edges.shape[1] == self.__times.shape[0], \
                f"The number of edges ({self.__edges.shape[1]}) and the length of edge times ({self.__times.shape[0]}) must match!"

    def read_edge_list(self, file_path, self_loops: bool = False):
        """
        Read the edge list file
        :param file_path: path of the file
        :param self_loops: whether to allow self loops or not
        """

        edges, edge_times, edge_weights = [], [], []
        self.__directed, self.__signed = False, False
        with open(file_path, 'r') as f:
            for line in f.readlines():

                # Discard the lines starting with #
                if line[0] == '#':
                    continue

                # Split the line into a list of strings
                tokens = re.split(';|,| |\t|\n', line.strip())

                if len(tokens) < 3:
                    raise Exception("An edge must consist of at least 3 columns: source, target and time!")

                # Add the edge
                edges.append((int(tokens[0]), int(tokens[1])))

                # Add the edge time
                edge_times.append(int(tokens[2]))

                # Add the edge weight if given
                if len(tokens) > 3:
                    edge_weights.append(float(tokens[3]))

                # If the first node of the edge is greater than the second, the graph is directed
                if edges[-1][0] > edges[-1][1]:
                    self.__directed = True

                # Check if the edge is a self loop
                if edges[-1][0] == edges[-1][1] and not self_loops:
                    raise ValueError("Self loops are not allowed!")

                # Check if the edge is signed
                if len(tokens) > 3 and edge_weights[-1] < 0:
                    self.__signed = True

        # Convert to torch format
        self.__edges = torch.as_tensor(edges, dtype=torch.long).T
        self.__times = torch.as_tensor(edge_times, dtype=torch.long)  # Unix timestamp
        self.__weights = torch.as_tensor(edge_weights, dtype=torch.float)

        # Get the nodes
        nodes = torch.unique(self.__edges)
        self.__nodes_num = len(nodes)

        # Check the minimum and maximum node labels
        assert min(nodes) == 0,\
            f"The nodes must start from 0, min node: {min(nodes)}."
        assert max(nodes) + 1 == len(nodes), \
            f"The max node label is {max(nodes)} but there are {len(nodes)} nodes so some nodes do not have any link."

        # Sort the edges
        sorted_indices = self.__times.argsort()
        self.__edges = self.__edges[:, sorted_indices]
        self.__times = self.__times[sorted_indices]
        self.__weights = self.__weights[sorted_indices] if len(edge_weights) else None

        # If the minimum and maximum time are not given, set them
        self.__init_time = self.__times.min()
        self.__last_time = self.__times.max()

        if self.__verbose:
            self.print_info()

    def has_isolated_nodes(self):
        """
        Check if the graph has isolated nodes
        """

        # Get the isolated nodes in the first split
        non_isolated_nodes = torch.sort(torch.unique(self.__edges))[0]

        return self.__nodes_num - len(non_isolated_nodes)

    def is_directed(self):
        """
        Check if the graph is directed
        """

        return self.__directed

    def is_signed(self):
        """
        Check if the graph is signed
        """

        return self.__signed

    def get_nodes_num(self) -> int:
        """
        Get the number of nodes
        """
        return self.__nodes_num

    def get_init_time(self) -> float:
        """
        Get the iniitial time
        """
        return self.__init_time

    def set_init_time(self, init_time: float):
        """
        Set the initial time
        """
        self.__init_time = init_time

    def get_last_time(self) -> float:
        """
        Get the last time
        """
        return self.__last_time

    def set_last_time(self, last_time: float):
        """
        Set the last time
        """
        self.__last_time = last_time

    def get_edges(self, idx: int = None) -> torch.Tensor:
        """
        Get the edges
        """
        if idx is None:
            return self.__edges
        else:
            return self.__edges[idx]

    def get_times(self) -> torch.Tensor:
        """
        Get the edge times
        """

        return self.__times

    def get_weights(self) -> torch.Tensor:
        """
        Get the states
        """

        if self.__weights is None:
            return torch.ones(self.__edges.shape[1], dtype=torch.float)
        else:
            return self.__weights

    def get_data(self, weights=False):
        """"
        Get all data
        :param weights: if True, return the weights
        :return: edges, times, states (if states=True)
        """

        if weights:
            return self.get_edges(), self.get_times(), self.get_weights()
        else:
            return self.get_edges(), self.get_times()

    def get_data_dict(self, weights: bool = False):
        """
        Get the data in a dictionary format
        :param weights: if True, return the weights
        :return: a dictionary with keys the source nodes and values a dictionary with keys the target nodes and values
        """
        data_dict = {}
        for i, j, t, w in zip(self.get_edges(0), self.get_edges(1), self.get_times(), self.get_weights()):

            if i.item() in data_dict:
                if j.item() in data_dict[i.item()]:
                    data_dict[i.item()][j.item()].append((t, w) if weights else t)
                else:
                    data_dict[i.item()][j.item()] = [(t, w) if weights else t]
            else:
                data_dict[i.item()] = {j.item(): [(t, w) if weights else t]}

        return data_dict

    def print_info(self):
        """
        Print the dataset info
        """
        print(f"+ Dataset information")
        print(f"\t- Number of nodes: {self.__nodes_num}")
        print(f"\t- Number of edges: {self.__edges.shape[1]}")
        print(f"\t- Is directed: {self.__directed}")
        print(f"\t- Is signed: {self.__signed}")
        print(f"\t- Initial time: {self.__init_time}")
        print(f"\t- Last time: {self.__last_time}")