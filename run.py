import torch
import pickle
from argparse import ArgumentParser, RawTextHelpFormatter
from src.learning import LearningModel
from src.dataset import Dataset
from utils.common import set_seed

# Global control for device
CUDA = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if (CUDA) and (device == "cuda:0"):
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def parse_arguments():
    parser = ArgumentParser(description="Examples: \n",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--dataset', type=str, required=True, help='Path of the dataset'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path of the model'
    )
    parser.add_argument(
        '--init_path', type=str, required=False, default="", help='A model path for the initialization'
    )
    parser.add_argument(
        '--mask_path', type=str, required=False, default="", help='Path of the file storing node pairs for masking'
    )
    parser.add_argument(
        '--log', type=str, required=False, default=None, help='Path of the log file'
    )
    parser.add_argument(
        '--bins_num', type=int, default=100, required=False, help='Number of bins'
    )
    parser.add_argument(
        '--dim', type=int, default=2, required=False, help='Dimension size'
    )
    parser.add_argument(
        '--last_time', type=float, default=1.0, required=False, help='The last time point of the training dataset'
    )
    parser.add_argument(
        '--k', type=int, default=10, required=False, help='Latent dimension size of the prior element'
    )
    parser.add_argument(
        '--prior_lambda', type=float, default=1e5, required=False, help='Scaling coefficient of the covariance'
    )
    parser.add_argument(
        '--epoch_num', type=int, default=100, required=False, help='Number of epochs'
    )
    parser.add_argument(
        '--spe', type=int, default=1, required=False, help='Number of steps per epoch'
    )
    parser.add_argument(
        '--batch_size', type=int, default=0, required=False, help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=0.1, required=False, help='Learning rate'
    )
    parser.add_argument(
        '--seed', type=int, default=19, required=False, help='Seed value to control the randomization'
    )
    parser.add_argument(
        '--verbose', type=bool, default=1, required=False, help='Verbose'
    )

    return parser.parse_args()


def process(args):

    dataset_path = args.dataset
    model_path = args.model_path
    init_path = args.init_path
    mask_path = args.mask_path
    log_file_path = args.log

    bins_num = args.bins_num
    dim = args.dim
    last_time = args.last_time
    K = args.k
    prior_lambda = args.prior_lambda
    epoch_num = args.epoch_num
    steps_per_epoch = args.spe
    batch_size = args.batch_size
    learning_rate = args.lr

    seed = args.seed
    verbose = args.verbose

    # Set the seed
    set_seed(seed=seed)

    # Load the dataset
    dataset = Dataset(path=dataset_path, normalize=False, verbose=verbose, seed=seed)
    if batch_size <= 0:
        batch_size = dataset.number_of_nodes()

    # Get the number of nodes
    nodes_num = dataset.number_of_nodes()
    data = dataset.get_pairs(), dataset.get_events()

    assert dataset.get_min_event_time() >= 0 and dataset.get_max_event_time() <= 1.0, \
        "The dataset contains events smaller than 0 or greater than 1.0!"

    # Load the node pairs for masking if given
    masked_pairs = None
    if mask_path != "":
        with open(mask_path, 'rb') as f:
            masked_pairs = torch.as_tensor(pickle.load(f), dtype=torch.int), #device=torch.device(device))

    if verbose:
        print(f"- The active device is {device}.")

    # Run/set the model
    if init_path == "":

        lm = LearningModel(
            data=data, nodes_num=nodes_num, bins_num=bins_num, dim=dim, last_time=last_time,
            prior_k=K, prior_lambda=prior_lambda, masked_pairs=masked_pairs,
            learning_rate=learning_rate, batch_size=batch_size, epoch_num=epoch_num, steps_per_epoch=steps_per_epoch,
            device=torch.device(device), verbose=verbose, seed=seed
        )

    else:

        kwargs, lm_state = torch.load(init_path, map_location=torch.device(device))
        lm = LearningModel(**kwargs)
        lm.load_state_dict(lm_state)

    # Learn the embeddings
    lm.learn(loss_file_path=log_file_path)

    # Save the model
    lm.save(model_path)


if __name__ == "__main__":
    args = parse_arguments()
    process(args)