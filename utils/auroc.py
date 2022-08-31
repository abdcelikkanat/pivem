import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, model_selection, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn import preprocessing
import pickle as pkl
from sklearn.metrics import precision_recall_curve
from scipy.stats import spearmanr

from scipy.spatial import distance

__OPS_LIST = ["cosine", "hadamard", "average", "l1", "l2"]

def split_into_training_test_sets(g, test_set_ratio, subsampling_ratio=0, remove_size=1000):
    print("--> The number of nodes: {}, the number of edges: {}".format(g.number_of_nodes(), g.number_of_edges()))

    print("+ Getting the gcc of the original graph.")
    # Keep the original graph
    train_g = g.copy()
    train_g.remove_edges_from(nx.selfloop_edges(train_g))  # remove self loops
    train_g = train_g.subgraph(max(nx.connected_components(train_g), key=len))
    if nx.is_frozen(train_g):
        train_g = nx.Graph(train_g)
    print("\t- Completed!")

    num_of_nodes = train_g.number_of_nodes()
    nodelist = list(train_g.nodes())
    edges = list(train_g.edges())
    num_of_edges = train_g.number_of_edges()
    print("--> The number of nodes: {}, the number of edges: {}".format(num_of_nodes, num_of_edges))

    if subsampling_ratio != 0:
        print("+ Subsampling initialization.")
        subsample_size = subsampling_ratio * num_of_nodes
        while (subsample_size < train_g.number_of_nodes()):
            chosen = np.random.choice(list(train_g.nodes()), size=remove_size)
            train_g.remove_nodes_from(chosen)
            train_g = train_g.subgraph(max(nx.connected_components(train_g), key=len))

            if nx.is_frozen(train_g):
                train_g = nx.Graph(train_g)

    print("+ Relabeling.")
    node2newlabel = {node: str(nodeIdx) for nodeIdx, node in enumerate(train_g.nodes())}
    train_g = nx.relabel_nodes(G=train_g, mapping=node2newlabel, copy=True)
    print("\t- Completed!")

    nodelist = list(train_g.nodes())
    edges = list(train_g.edges())
    num_of_nodes = train_g.number_of_nodes()
    num_of_edges = train_g.number_of_edges()
    print("--> The of nodes: {}, the number of edges: {}".format(num_of_nodes, num_of_edges))

    print("+ Splitting into train and test sets.")
    test_size = int(test_set_ratio * num_of_edges)

    test_g = nx.Graph()
    test_g.add_nodes_from(nodelist)

    count = 0
    idx = 0
    perm = np.arange(num_of_edges)
    while (count < test_size and idx < num_of_edges):
        if count % 10000 == 0:
            print("{}/{}".format(count, test_size))
        # Remove the chosen edge
        chosen_edge = edges[perm[idx]]
        train_g.remove_edge(chosen_edge[0], chosen_edge[1])
        if chosen_edge[1] in nx.connected._plain_bfs(train_g, chosen_edge[0]):
            test_g.add_edge(chosen_edge[0], chosen_edge[1])
            count += 1
        else:
            train_g.add_edge(chosen_edge[0], chosen_edge[1])

        idx += 1
    if idx == num_of_edges:
        raise ValueError("There are no enough edges to sample {} number of edges".format(test_size))
    else:
        print("--> Completed!")

    if count != test_size:
        raise ValueError("Enough positive edge samples could not be found!")

    # Generate the negative samples
    print("\+ Generating negative samples")
    count = 0
    negative_samples_idx = [[] for _ in range(num_of_nodes)]
    negative_samples = []
    while count < 2 * test_size:
        if count % 10000 == 0:
            print("{}/{}".format(count, 2 * test_size))
        uIdx = np.random.randint(num_of_nodes - 1)
        vIdx = np.random.randint(uIdx + 1, num_of_nodes)

        if vIdx not in negative_samples_idx[uIdx]:
            negative_samples_idx[uIdx].append(vIdx)

            u = nodelist[uIdx]
            v = nodelist[vIdx]

            negative_samples.append((u, v))

            count += 1

    train_neg_samples = negative_samples[:test_size]
    test_neg_samples = negative_samples[test_size:test_size * 2]

    return train_g, test_g, train_neg_samples, test_neg_samples


############################################################################################################


def read_emb_file(file_path):

    with open(file_path, 'r') as fin:
        # Read the first line
        num_of_nodes, dim = (int(token) for token in fin.readline().strip().split())

        # read the embeddings
        embs = [[] for _ in range(num_of_nodes)]

        for line in fin.readlines():
            tokens = line.strip().split()
            embs[int(tokens[0])] = [float(v) for v in tokens[1:]]

        embs = np.asarray(embs, dtype=np.float)

    return embs


def extract_feature_vectors_from_embeddings(edges, embeddings, binary_operator):
    features = []
    for i in range(len(edges)):
        edge = edges[i]
        vec1 = embeddings[int(edge[0])]
        vec2 = embeddings[int(edge[1])]

        if binary_operator == "cosine":
            value = 1.0 - distance.cosine(vec1, vec2)
        elif binary_operator == "hadamard":
            value = [vec1[i] * vec2[i] for i in range(len(vec1))]
        elif binary_operator == "average":
            value = 0.5 * (vec1 + vec2)
        elif binary_operator == "l1":
            value = abs(vec1 - vec2)
        elif binary_operator == "l2":
            value = abs(vec1 - vec2) ** 2
        else:
            raise ValueError("Invalid operator!")

        features.append(value)

    features = np.asarray(features)
    # Reshape the feature vector if it is 1d vector
    if binary_operator == "cosine":
        features = features.reshape(-1, 1)

    return features


################################################################################################

def predict(dataset_folder_path, emb_file_path, output_path):
    print("-----------------------------------------------")
    print("Input folder: {}".format(dataset_folder_path))
    print("Emb path: {}".format(emb_file))
    print("-----------------------------------------------")

    # Load sample files
    with open(os.path.join(dataset_folder_path, "samples.pkl"), 'rb') as f:
        samples_data = pkl.load(f)
        valid_labels = samples_data["valid_labels"]
        valid_samples = samples_data["valid_samples"]
        test_labels = samples_data["test_labels"]
        test_samples = samples_data["test_samples"]

    f = open(output_path, 'w')

    embs = read_emb_file(emb_file_path)

    samples = test_samples
    labels = test_labels
    test_intervals_num = len(test_labels)

    pred_scores = {op_name: [[] for _ in range(test_intervals_num)] for op_name in __OPS_LIST}
    # Predicted scores
    for op_name in __OPS_LIST:

        f.write(f"Op_name: {op_name}\n")

        for b in range(test_intervals_num):
            # for sample in samples[b]:
            #     i, j, t_init, t_last = sample

            bin_edges = [[edge[0], edge[1]] for edge in samples[b]]
            test_features = extract_feature_vectors_from_embeddings(
                edges=bin_edges, embeddings=embs, binary_operator=op_name
            )
            clf = LogisticRegression()
            clf.fit(test_features, labels[b])
            test_preds = clf.predict_proba(test_features)[:, 1]

            roc_auc = roc_auc_score(y_true=labels[b], y_score=test_preds)
            f.write(f"Roc AUC, Bin Id {b}: {roc_auc}\n")
            pr_auc = average_precision_score(y_true=labels[b], y_score=test_preds)
            f.write(f"PR AUC, Bin Id {b}: {pr_auc}\n")
            f.write("")

        edges = [[edge[0], edge[1]] for bin_samples in samples for edge in bin_samples ]
        labels = [l for bin_labels in labels for l in bin_labels]
        test_features = extract_feature_vectors_from_embeddings(
            edges=edges, embeddings=embs, binary_operator=op_name
        )
        clf = LogisticRegression()
        clf.fit(test_features, labels)
        test_preds = clf.predict_proba(test_features)[:, 1]

        roc_auc_complete = roc_auc_score(
            y_true=labels,
            y_score=test_preds
        )
        f.write(f"Roc AUC in total: {roc_auc_complete}\n")

        pr_auc_complete = average_precision_score(
            y_true=labels,
            y_score=test_preds
        )
        f.write(f"PR AUC in total: {pr_auc_complete}\n")

    # test_results = {op_name: 0. for op_name in __OPS_LIST}
    # valid_results = {op_name: 0. for op_name in __OPS_LIST}
    # test_corr = {op_name: 0. for op_name in __OPS_LIST}
    # valid_corr = {op_name: 0. for op_name in __OPS_LIST}
    #
    # for op_name in __OPS_LIST:
    #     test_features = extract_feature_vectors_from_embeddings(
    #         edges=test_samples, embeddings=embs, binary_operator=op_name
    #     )
    #     clf = LogisticRegression()
    #     clf.fit(test_features, test_labels)
    #     test_preds = clf.predict_proba(test_features)[:, 1]
    #     test_roc = roc_auc_score(y_true=test_labels, y_score=test_preds)
    #     test_results[op_name] = test_roc
    #     test_corr[op_name] = spearmanr(np.asarray(test_labels, dtype=np.float), np.asarray(test_preds, dtype=np.float))
    #
    #     valid_features = extract_feature_vectors_from_embeddings(
    #         edges=test_samples, embeddings=embs, binary_operator=op_name
    #     )
    #     clf = LogisticRegression()
    #     clf.fit(valid_features, valid_labels)
    #     valid_preds = clf.predict_proba(valid_features)[:, 1]
    #     valid_roc = roc_auc_score(y_true=valid_labels, y_score=valid_preds)
    #     valid_results[op_name] = valid_roc
    #     valid_corr[op_name] = spearmanr(np.asarray(test_labels, dtype=np.float), np.asarray(valid_preds, dtype=np.float))
    #
    # with open(output_path, 'w') as f:
    #     f.write("Valid results\n")
    #     for op_name in __OPS_LIST:
    #         f.write(f"{op_name}: {valid_results[op_name]} {valid_corr[op_name]}\n")
    #
    #     f.write("Test results\n")
    #     for op_name in __OPS_LIST:
    #         f.write(f"{op_name}: {test_results[op_name]} {test_corr[op_name]}\n")


if sys.argv[1] == 'predict':

    dataset_folder_path = sys.argv[2]
    emb_file = sys.argv[3]
    output_path = sys.argv[4]

    predict(dataset_folder_path, emb_file, output_path)
