import os
import torch
from argparse import ArgumentParser, RawTextHelpFormatter
from sklearn.metrics import average_precision_score, roc_auc_score
from src.learning import LearningModel
import pickle as pkl

########################################################################################################################
parser = ArgumentParser(description="Examples: \n", formatter_class=RawTextHelpFormatter)
parser.add_argument(
    '--samples_folder', type=str, required=True, help='Path of the samples folder'
)
parser.add_argument(
    '--model_path', type=str, required=True, help='Path of the model'
)
parser.add_argument(
    '--pred_ratio', type=float, default=0.1, required=True, help='Prediction ratio'
)
parser.add_argument(
    '--output_path', type=str, required=True, help='Path of the output file'
)

args = parser.parse_args()
########################################################################################################################

seed = 19
# Set some parameters
samples_folder = args.samples_folder
model_path = args.model_path
pred_ratio = args.pred_ratio
output_path = args.output_path

train_last_time = 1.0 - pred_ratio

########################################################################################################################

print("+ Model is being read...")
# Load the model
kwargs, lm_state = torch.load(model_path, map_location=torch.device('cpu'))
kwargs['device'] = 'cpu'
lm = LearningModel(**kwargs)
lm.load_state_dict(lm_state)
print("\t- Completed.")

########################################################################################################################

with open(os.path.join(samples_folder, "pos.samples"), 'rb') as f:
    pos_samples = pkl.load(f)
with open(os.path.join(samples_folder, "neg.samples"), 'rb') as f:
    neg_samples = pkl.load(f)

labels = [1]*len(pos_samples) + [0]*len(neg_samples)
samples = pos_samples + neg_samples

# print(pos_samples)
########################################################################################################################

pred_scores = []
for sample in samples:

    pred_scores.append(
        lm.get_intensity_integral_for(
            i=sample[0], j=sample[1],
            interval=torch.as_tensor([0., train_last_time])
        ).detach().numpy()
    )


with open(output_path, 'w') as f:
    roc_auc = roc_auc_score(y_true=labels, y_score=pred_scores)
    f.write(f"Roc_AUC: {roc_auc}\n")
    pr_auc = average_precision_score(y_true=labels, y_score=pred_scores)
    f.write(f"PR_AUC: {pr_auc}\n")
