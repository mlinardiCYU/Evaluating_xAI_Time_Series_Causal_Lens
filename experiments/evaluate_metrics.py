import os
import json
import argparse

import pandas as pd
import numpy as np

from collections import defaultdict

import scipy.stats
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from imblearn.over_sampling import RandomOverSampler

import networkx as nx

rng = np.random.default_rng(0)  # random seed
legacy_rng = np.random.RandomState(0)

rootdir = "../"

################
#
#   Opening config file
#
################

parser = argparse.ArgumentParser()
parser.add_argument('filename', help="The name of the json configuration file, as in ./configs/name.json")
args = parser.parse_args()

with open("./configs/" + args.filename + ".json", "r") as jsonfile:
    data_config = json.load(jsonfile)

################
#
#   Model constructors import (define the wrappers in another file for clarity)
#
################ 

if data_config["modelname"] == "dCAM":
    from model_wrappers import dCAM_model_constructor as model_constructor
    from model_wrappers import device

    model_constructor_config = data_config["model_parameters"]
    model_constructor_config["device"] = device
    model_constructor_config["original_length"] = data_config['pastPointsToForecast']
    model_constructor_config["num_classes"] = data_config['numberLabels']


    def constructor_from_nbvar(nbvar):
        model_constructor_config["original_dim"] = nbvar
        return model_constructor_config


elif data_config["modelname"] == "XCM_pytorch":
    from model_wrappers import XCM_model_constructor as model_constructor
    from model_wrappers import device
    
    model_constructor_config = data_config["model_parameters"]
    model_constructor_config["device"] = device
    model_constructor_config["input_shape"] = [data_config['pastPointsToForecast'], 0, 1]
    model_constructor_config["n_class"] = data_config['numberLabels']


    def constructor_from_nbvar(nbvar):
        model_constructor_config["input_shape"][1] = nbvar
        return model_constructor_config
    
elif data_config["modelname"] == "transformer":
    from model_wrappers import Transformer_model_constructor as model_constructor
    from model_wrappers import device

    model_constructor_config = data_config["model_parameters"]
    model_constructor_config["seq_len"]=data_config['pastPointsToForecast']
    model_constructor_config["num_classes"] = data_config['numberLabels']
    model_constructor_config["device"] = device

    def constructor_from_nbvar(nbvar):
        model_constructor_config["input_dim"]=nbvar
        return model_constructor_config

elif data_config["modelname"] == "LSTM":
    from model_wrappers import LSTM_model_constructor as model_constructor
    from model_wrappers import device

    model_constructor_config = data_config["model_parameters"]
    model_constructor_config["device"] = device
    model_constructor_config["original_length"] = data_config['pastPointsToForecast']
    model_constructor_config["num_classes"] = data_config['numberLabels']


    def constructor_from_nbvar(nbvar):
        model_constructor_config["input_size"] = nbvar
        return model_constructor_config

else:
    raise Exception("Specified modelname argument of config file not known")

################
#
#   Training loop routine for restricted models
#
################


def train_restricted_model(restricted_window_data, window_labels, fccv=False):
    model_constructor_restricted_config = constructor_from_nbvar(restricted_window_data.shape[2])

    #   separate into folds

    if fccv:  # forward chain cross validation (3 folds) on test dataset...
        folds = 3
        skf_generator = []
        for i in range(folds, 0, -1):
            separation = int((1 - (data_config['test_split_fraction']*i)/folds) * len(restricted_window_data))
            end = int(len(restricted_window_data) * (1 - (data_config['test_split_fraction'] * (i-1)) / folds))
            train_indices = list(range(separation))
            test_indices = list(range(separation, end))
            skf_generator.append((train_indices, test_indices))

    elif data_config['numberFolds'] > 1:
        skf_object = sklearn.model_selection.StratifiedKFold(data_config['numberFolds'], shuffle=True,
                                                             random_state=np.random.RandomState(0))
        skf_generator = skf_object.split(restricted_window_data, window_labels)

    elif data_config['numberFolds'] == 1:
        # take the earlier 1-test_split_fraction percent of the data as training, and the latter as test.
        separation = int((1 - data_config['test_split_fraction']) * len(restricted_window_data))
        train_indices = list(range(separation))
        test_indices = list(range(separation, len(restricted_window_data)))
        skf_generator = [(train_indices, test_indices)]
    else:
        raise (Exception("Invalid configuration file field <numberFolds>, specify an integer >= 1."))

    #   Train, predict, explain - loop

    restricted_y_true = []
    restricted_y_pred = []
    restricted_y_score = []

    for ith_fold, (train_index, test_index) in enumerate(skf_generator):
        data_train = restricted_window_data[train_index]
        labels_train = window_labels[train_index]
        data_test = restricted_window_data[test_index]
        labels_test = window_labels[test_index]

        # insert oversampling here
        new_index, labels_train = RandomOverSampler(random_state=legacy_rng).fit_resample(
            np.arange(len(data_train)).reshape((-1, 1)), labels_train)
        data_train = data_train[new_index.reshape((-1,))]

        # insert model declaration here
        model = model_constructor(**model_constructor_restricted_config)

        # insert model training here
        model.train(data_train, labels_train)  # replace as necessary

        # obtain predicted labels from the test set
        score_pred, labels_pred = model.predict(data_test)  # replace as necessary

        # keep track of results
        restricted_y_pred.append(labels_pred)
        restricted_y_true.append(labels_test)
        restricted_y_score.append(score_pred)

    # convert to array
    restricted_y_true = np.concatenate(restricted_y_true)
    restricted_y_pred = np.concatenate(restricted_y_pred)
    restricted_y_score = np.concatenate(restricted_y_score)

    # compute metrics
    # accuracy = sklearn.metrics.accuracy_score(y_true=restricted_y_true,y_pred=restricted_y_pred)
    # balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true=restricted_y_true,y_pred=restricted_y_pred)
    # f1_score = sklearn.metrics.f1_score(y_true=restricted_y_true,y_pred=restricted_y_pred)

    if data_config['numberLabels'] == 2:
        roc_auc_score = sklearn.metrics.roc_auc_score(restricted_y_true, restricted_y_score)
    else:
        roc_auc_score = sklearn.metrics.roc_auc_score(restricted_y_true, restricted_y_score, multi_class="ovo")

    # if data_config['numberLabels']==2:
    #    average_precision_score = sklearn.metrics.average_precision_score(restricted_y_true,restricted_y_score)
    # else:
    #    average_precision_score = None

    # tn,fp,fn,tp = sklearn.metrics.confusion_matrix(y_true=restricted_y_true,
    #                                                 y_pred=restricted_y_pred,normalize="true").ravel()

    # update dataframe
    new_row = {"config_name": args.filename, "dataset_name": data_config['dataset_name'],
               "model": data_config['modelname'],
               "roc_auc_score": roc_auc_score
               # "average_precision_score":average_precision_score,
               # "accuracy":accuracy,"balanced_accuracy":balanced_accuracy,"f1_score":f1_score,
               # "tn":tn,"fp":fp,"fn":fn,"tp":tp
               }

    return new_row


################
#
#   Iterations over files
#
################

for filename in os.listdir(rootdir + "/data/" + data_config["dataset_name"] + "/"):
    if not os.path.isfile(rootdir + "/data/" + data_config["dataset_name"] + "/" + filename):
        continue

    print("File", filename, "is beginning.")

    ################
    #
    #   Result dataframes
    #
    ################

    df_result_classif = pd.DataFrame()
    df_result_restrict_classif_true = pd.DataFrame()
    df_result_restrict_classif_pred = pd.DataFrame()
    df_result_expl = pd.DataFrame()

    ################
    #
    #   Opening data
    #
    ################

    if data_config["dataset_name"][:11] == "SynthNonlin":
        df = pd.read_csv(rootdir + "/data/" + data_config["dataset_name"] + "/" + filename)
        df = df[df.columns[1:]]

    elif data_config["dataset_name"][:4] == "fMRI":
        df = pd.read_csv(rootdir + "/data/" + data_config["dataset_name"] + "/" + filename)

    elif data_config["dataset_name"][:10] == "FinanceCPT":
        df = pd.read_csv(rootdir + "/data/" + data_config["dataset_name"] + "/" + filename, header=None)
        df.columns = [str(i) for i in df.columns]

    elif data_config["dataset_name"][:18] == "TestCLIM_N-5_T-250":
        df = pd.read_csv(rootdir + "/data/" + data_config["dataset_name"] + "/" + filename, header=None, sep=" ")
        df.columns = [str(i) for i in df.columns]
    else:
        raise Exception("Dataset specified in config file is not implemented")

    VAR_NAMES = list(df.columns)

    if data_config["dataset_name"][:11] == "SynthNonlin":
        if data_config["dataset_name"] == "SynthNonlin/7ts2h":
            ground_truth_parents = defaultdict(list)
            ground_truth_lags = data_config['pastPointsToForecast']
            ground_truth_parents["A"] = [("D", 1), ("A", 1)] + [("B", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["D"] = [("H", 1), ("D", 1)] + [("E", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["H"] = [("C", 1), ("H", 1)]
            ground_truth_parents["C"] = [("C", 1)]
            ground_truth_parents["F"] = [("C", 1), ("F", 1)]
            ground_truth_parents["B"] = [("F", 1), ("B", 1)] + [("A", i) for i in range(1, ground_truth_lags + 1)]
            ground_truth_parents["E"] = [("B", 1), ("E", 1)] + [("D", i) for i in range(1, ground_truth_lags + 1)]
        else:
            raise Exception("Dataset specified in config file is not implemented")

    elif data_config["dataset_name"][:4] == "fMRI":
        index = filename[10:]
        index = index[:-4]
        g_truth_name = "fMRI_processed_by_Nauta/ground_truth/sim" + index + "_gt_processed.csv"

        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 0
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            ground_truth_lags = max(ground_truth_lags, lag)

    elif data_config["dataset_name"][:10] == "FinanceCPT":

        g_truth_name = "FinanceCPT/ground_truth/" + filename[:filename.find("_returns")] + ".csv"
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 0
        for cause, effect, lag in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), lag))
            ground_truth_lags = max(ground_truth_lags, lag)

    elif data_config["dataset_name"][:18] == "TestCLIM_N-5_T-250":
        g_truth_name = "TestCLIM_N-5_T-250/estimated_ground_truth/" + filename
        df_truth = pd.read_csv(rootdir + "data/" + g_truth_name, header=None, sep=",")
        ground_truth_parents = defaultdict(list)
        ground_truth_lags = 2
        for cause, effect in df_truth.values:
            ground_truth_parents[str(effect)].append((str(cause), 1))
            ground_truth_parents[str(effect)].append((str(cause), 2))
    else:
        raise Exception("Dataset specified in config file is not implemented")

    ################
    #
    #   Creating the causal graphs
    #
    ################

    node_names = dict()
    for var in VAR_NAMES:
        for lag in range(ground_truth_lags + 1):
            node_names[(var, lag)] = "L" + str(lag) + "." + str(var)

    # graph with only parents
    ground_truth_graph = nx.DiGraph()
    for key in node_names:
        ground_truth_graph.add_node(node_names[key])
    for key in ground_truth_parents:
        child_name = "L0." + str(key)
        for parent in ground_truth_parents[key]:
            parent_name = "L" + str(parent[1]) + "." + str(parent[0])
            ground_truth_graph.add_edge(parent_name, child_name)

    # summary graph (no lags)
    summary_graph = nx.DiGraph()
    summary_graph.add_nodes_from(VAR_NAMES)
    for cause, effect in ground_truth_graph.edges:
        lag = cause[1:cause.find(".")]
        cause = cause[cause.find(".") + 1:]
        effect = effect[effect.find(".") + 1:]
        if not summary_graph.has_edge(cause, effect):
            summary_graph.add_edge(cause, effect, lags=[lag])
        else:
            summary_graph[cause][effect]["lags"].append(lag)

    # window graph adapted to pastPointsToForecast
    window_graph = nx.DiGraph()
    node_names = []
    for var in VAR_NAMES:
        for lag in range(0, data_config['pastPointsToForecast'] + 1):
            node_names.append("L" + str(lag) + "." + str(var))
    window_graph.add_nodes_from(node_names)
    for cause, effect in ground_truth_graph.edges:
        lag = int(cause[1:cause.find(".")])
        cause = cause[cause.find(".") + 1:]
        effect = effect[effect.find(".") + 1:]
        for L in range(lag, data_config['pastPointsToForecast'] + 1):
            window_graph.add_edge("L" + str(L) + "." + cause, "L" + str(L - lag) + "." + effect)
    window_graph_pos = dict([(node,
                              (1 - int(node[1:node.find(".")]) / data_config['pastPointsToForecast'],
                               1 - VAR_NAMES.index(node[node.find(".") + 1:]) / len(VAR_NAMES)))
                             for node in window_graph.nodes()])

    # extractor functions

    def get_all_parents(graph, target):
        if "L0." + target in graph.nodes:
            return list(graph.predecessors("L0." + target))
        else:
            return list(graph.predecessors(target))


    def get_all_ancestors(graph, target):
        if "L0." + target in graph.nodes:
            return list(nx.ancestors(graph, "L0." + target))
        else:
            return list(nx.ancestors(graph, target))


    def get_all_connected(graph, target):
        if "L0." + target in graph.nodes:
            return list(nx.node_connected_component(graph.to_undirected(), "L0." + target))
        else:
            return list(nx.node_connected_component(graph.to_undirected(), target))


    ################
    #
    #   Iterate over target variable
    #
    ################

    for TARGET_NAME in VAR_NAMES:
        print("\tVariable", TARGET_NAME, "is beginning.")

        ################
        #
        #   Creating the list of reference causal variables
        #
        ################

        if data_config['causeExtraction'] == "parents":
            LAGGED_CAUSAL_LIST = get_all_parents(window_graph, TARGET_NAME)
            NONLAGGED_CAUSAL_LIST = get_all_parents(summary_graph, TARGET_NAME)
        elif data_config['causeExtraction'] == "ancestors":
            LAGGED_CAUSAL_LIST = get_all_ancestors(window_graph, TARGET_NAME)
            NONLAGGED_CAUSAL_LIST = get_all_ancestors(summary_graph, TARGET_NAME)
        elif data_config['causeExtraction'] == "connected":
            LAGGED_CAUSAL_LIST = get_all_connected(window_graph, TARGET_NAME)
            NONLAGGED_CAUSAL_LIST = get_all_connected(summary_graph, TARGET_NAME)
        else:
            raise Exception("causeExtraction method specified in config file is not implemented")

        if len(NONLAGGED_CAUSAL_LIST) == 0:
            print("\t\tThe variable has no cause: excluded.")
            continue

        ################
        #
        #   Use k-means clustering to create labels, separate into time windows
        #
        ################

        data_for_labels = df[TARGET_NAME].values.reshape((-1, 1))

        est = KBinsDiscretizer(n_bins=data_config['numberLabels'], encode='ordinal', strategy='kmeans',
                               random_state=np.random.RandomState(0))
        est.fit(data_for_labels)
        labels = est.transform(data_for_labels)
        labels = labels.reshape((-1,))

        if data_config["minority_only"]:
            if data_config["numberLabels"] == 2:
                minority_class = int(np.count_nonzero(labels == 0) > len(labels) // 2)  # ONLY WORKS FOR BINARY LABELS
            else:
                raise (Exception("Cannot use the setting minority_only with numberLabels higher than 2"))

        # Normalize data with z-normalization.

        df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

        # separate into windows

        pastPointsToForecast = data_config['pastPointsToForecast']
        window_data = []
        window_labels = []

        for windowbegin in range(df.values.shape[0] - pastPointsToForecast):
            current_window = df.values[windowbegin:windowbegin + pastPointsToForecast, :]
            current_label = labels[windowbegin + pastPointsToForecast]
            window_data.append(current_window)
            window_labels.append(current_label)

        window_data = np.array(window_data)
        window_labels = np.array(window_labels)

        # Verify that both classes are present in the training set.

        separation = int((1-data_config['test_split_fraction'])*len(window_labels))
        if np.count_nonzero(window_labels[:separation]==0) in [0,separation]:
           print("Nonstationary data for target variable: only one label in training data")
           continue

        ################
        #
        #   Use the Forward Chain Cross Validation to keep only models with sufficient roc_auc_score.
        #
        ################

        new_row = train_restricted_model(window_data, window_labels, fccv=True)
        if new_row["roc_auc_score"] < 0.7:
            print("\t\t ROCAUC score of main model is too low according to fccv, skipped")
            continue

        ################
        #
        #   Opening results for each explanation method
        #
        ################

        for explanation_method in data_config["explanation_methods"]:

            savefile = "./saved_explanations/" + data_config["dir_to_save"] + os.path.splitext(filename)[0] + "_" + \
                       TARGET_NAME + "_" + explanation_method + ".npz"
            npzobject = np.load(savefile)
            y_true, y_pred, y_score, explanations, times = npzobject["y_true"], npzobject["y_pred"], \
                npzobject["y_score"], npzobject["explanations"], npzobject["times"]

            ################
            #
            #   Classification metrics computation.
            #
            ################
            if data_config["minority_only"]:
                if np.sum((y_true == minority_class) * (y_pred == minority_class)) == 0:
                    print("\t\tNumber of minority class datapoints correctly identified:",
                          np.sum((y_true == minority_class) * (y_pred == minority_class)))
                    print("\t\tExplanation", explanation_method, "skipped.")
                    continue

            # compute metrics
            # accuracy = sklearn.metrics.accuracy_score(y_true=y_true,y_pred=y_pred)
            # balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true=y_true,y_pred=y_pred)
            # f1_score = sklearn.metrics.f1_score(y_true=y_true,y_pred=y_pred)

            if data_config['numberLabels'] == 2:
                roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_score)
            else:
                roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_score, multi_class="ovo")

            # if data_config['numberLabels']==2:
            #    average_precision_score = sklearn.metrics.average_precision_score(y_true,y_score)
            # else:
            #    average_precision_score = None

            # tn,fp,fn,tp = sklearn.metrics.confusion_matrix(y_true=y_true,y_pred=y_pred,normalize="true").ravel()

            # update dataframe
            new_row = {"config_name": args.filename, "dataset_name": data_config['dataset_name'],
                       "filename": filename, "target": TARGET_NAME,
                       "model": data_config['modelname'], "explanation": explanation_method,
                       "roc_auc_score": roc_auc_score
                       # "average_precision_score":average_precision_score,
                       # "accuracy":accuracy,
                       # "balanced_accuracy":balanced_accuracy,
                       # "f1_score":f1_score,
                       # "tn":tn,"fp":fp,"fn":fn,"tp":tp,"times":int(times)
                       }

            df_result_classif = pd.concat([df_result_classif, pd.DataFrame.from_records([new_row])])

            # if roc_auc_score is lower than 0.7, skip the rest
            if roc_auc_score < 0.7:
                print("\t\t ROCAUC score of main model is too low, skipped")
                continue

            ################
            #
            #   TopK transformation of the explanations: order by decreasing FI then iterate over K
            #
            ################

            # sort explanation array and transform to text format
            if explanation_method == "DM":
                indices = np.argsort(explanations.reshape((explanations.shape[0],explanations.shape[1], -1)))[:, :, ::-1]
                ordered_lagged = np.vectorize(lambda x: "L" + str(pastPointsToForecast - x % pastPointsToForecast) + "." +
                                                        VAR_NAMES[x // pastPointsToForecast])(indices)
                aggregated = np.sum(explanations, axis=-1)
                indices = np.argsort(aggregated)[:, :, ::-1]
                ordered_nonlagged = np.vectorize(lambda x: VAR_NAMES[x])(indices)
            else:
                indices = np.argsort(explanations.reshape((len(explanations), -1)))[:, ::-1]
                ordered_lagged = np.vectorize(lambda x: "L" + str(pastPointsToForecast - x % pastPointsToForecast) + "." +
                                                        VAR_NAMES[x // pastPointsToForecast])(indices)
                aggregated = np.sum(explanations, axis=-1)
                indices = np.argsort(aggregated)[:, ::-1]
                ordered_nonlagged = np.vectorize(lambda x: VAR_NAMES[x])(indices)

            # create initial consistency frequency list
            frequencies_lagged = [defaultdict(lambda: 0) for label in range(data_config['numberLabels'])]
            frequencies_nonlagged = [defaultdict(lambda: 0) for label in range(data_config['numberLabels'])]

            # iterate over K
            topKlagged = 0
            topKnonlagged = 0
            while topKlagged < data_config['topKlagged_max'] or topKnonlagged < data_config['topKnonlagged_max'] or \
                    topKlagged <= len(VAR_NAMES) * pastPointsToForecast or topKnonlagged <= len(VAR_NAMES):

                topKlagged += 1
                topKnonlagged += 1

                if explanation_method == "DM":
                    # Use the explanation of the mask corresponding to the smallest area for lagged values
                    # Masks are increasing order according to area size
                    topKlagged_list = ordered_lagged[:, topKlagged//data_config['topKlagged_max'], :topKlagged]
                    # Last mask for non lagged
                    topKnonlagged_list = ordered_nonlagged[:, -1, :topKnonlagged]
                else:
                    topKlagged_list = ordered_lagged[:, :topKlagged]
                    topKnonlagged_list = ordered_nonlagged[:, :topKnonlagged]

                ################
                #
                #   Consistency of the topK explanations, using only correctly classified examples
                #
                ################

                if topKlagged <= data_config['topKlagged_max']:
                    # add the last included feature number of occurrences
                    for i, topKexplanation in enumerate(topKlagged_list):
                        # only compute on correctly classified examples
                        if y_pred[i] == y_true[i]:
                            frequencies_lagged[y_pred[i]][topKexplanation[-1]] += 1

                    consistency_metric_lagged = []
                    for label in range(data_config['numberLabels']):
                        freq = np.array(list(frequencies_lagged[label].values()))
                        freq = freq / np.sum(freq)
                        consistency_metric_lagged.append(scipy.stats.entropy(freq, base=2))

                if topKnonlagged <= data_config['topKnonlagged_max']:
                    # add the last included feature number of occurrences
                    for i, topKexplanation in enumerate(topKnonlagged_list):
                        # only compute on correctly classified examples
                        if y_pred[i] == y_true[i]:
                            frequencies_nonlagged[y_pred[i]][topKexplanation[-1]] += 1

                    consistency_metric_nonlagged = []
                    for label in range(data_config['numberLabels']):
                        freq = np.array(list(frequencies_nonlagged[label].values()))
                        freq = freq / np.sum(freq)
                        consistency_metric_nonlagged.append(scipy.stats.entropy(freq, base=2))

                ################
                #
                #   Causal computation of precision@k, recall@k, using only correctly classified examples
                #
                ################

                if topKlagged <= data_config['topKlagged_max']:
                    precision_list_lagged = []
                    recall_list_lagged = []
                    for i, topKexplanation in enumerate(topKlagged_list):
                        if y_pred[i] != y_true[i]:
                            continue
                        if data_config["minority_only"] and y_pred[i] != minority_class:
                            continue
                        total = 0
                        for true_cause in LAGGED_CAUSAL_LIST:
                            if true_cause in topKexplanation:
                                total += 1
                        recall = total / len(LAGGED_CAUSAL_LIST)
                        precision = total / len(topKexplanation)
                        recall_list_lagged.append(recall)
                        precision_list_lagged.append(precision)

                if topKnonlagged <= data_config['topKnonlagged_max']:
                    precision_list_nonlagged = []
                    recall_list_nonlagged = []
                    for i, topKexplanation in enumerate(topKnonlagged_list):
                        if y_pred[i] != y_true[i]:
                            continue
                        if data_config["minority_only"] and y_pred[i] != minority_class:
                            continue
                        total = 0
                        for true_cause in NONLAGGED_CAUSAL_LIST:
                            if true_cause in topKexplanation:
                                total += 1
                        recall = total / len(NONLAGGED_CAUSAL_LIST)
                        precision = total / len(topKexplanation)
                        recall_list_nonlagged.append(recall)
                        precision_list_nonlagged.append(precision)

                ################
                #
                #   Computing a random selector method precision@k and recall@k
                #
                ################

                if topKlagged <= data_config['topKlagged_max']:
                    k, n, p = topKlagged, len(VAR_NAMES) * pastPointsToForecast, len(LAGGED_CAUSAL_LIST)
                    # build theoretical distribution of the number of true positive of a random selection method.
                    theoretical_distribution_lagged = []
                    for i in range(max(0, p + k - n), min(k, p) + 1):
                        # there are number_possibilities selections with i true positive
                        number_possibilities = scipy.special.comb(p, i) * scipy.special.comb(n - p, k - i)
                        theoretical_distribution_lagged.append((i, number_possibilities))
                    theoretical_distribution_lagged = np.array(theoretical_distribution_lagged)
                    theoretical_distribution_lagged[:, 1] /= scipy.special.comb(n, k)

                    random_precision_lagged = np.sum(np.prod(theoretical_distribution_lagged, axis=-1)) / k
                    random_recall_lagged = np.sum(np.prod(theoretical_distribution_lagged, axis=-1)) / p

                if topKnonlagged <= data_config['topKnonlagged_max']:
                    k, n, p = topKnonlagged, len(VAR_NAMES), len(NONLAGGED_CAUSAL_LIST)
                    # build theoretical distribution of the number of true positive of a random selection method.
                    theoretical_distribution_nonlagged = []
                    for i in range(max(0, p + k - n), min(k, p) + 1):
                        # there are number_possibilities selections with i true positive
                        number_possibilities = scipy.special.comb(p, i) * scipy.special.comb(n - p, k - i)
                        theoretical_distribution_nonlagged.append((i, number_possibilities))
                    theoretical_distribution_nonlagged = np.array(theoretical_distribution_nonlagged)
                    theoretical_distribution_nonlagged[:, 1] /= scipy.special.comb(n, k)

                    random_precision_nonlagged = np.sum(np.prod(theoretical_distribution_nonlagged, axis=-1)) / k
                    random_recall_nonlagged = np.sum(np.prod(theoretical_distribution_nonlagged, axis=-1)) / p

                ################
                #
                #   Computing global metrics precision@k and recall@k, with correspondence with random selector.
                #
                ################

                if topKlagged <= data_config['topKlagged_max']:
                    precision_mean_lagged = np.mean(precision_list_lagged)
                    recall_mean_lagged = np.mean(recall_list_lagged)
                    # precision_std_lagged = np.std(precision_list_lagged)
                    # recall_std_lagged = np.std(recall_list_lagged)
                    # precision_median_lagged = np.median(precision_list_lagged)
                    # recall_median_lagged = np.median(recall_list_lagged)
                    # precision_iqr_lagged = scipy.stats.iqr(precision_list_lagged)
                    # recall_iqr_lagged = scipy.stats.iqr(recall_list_lagged)

                    # statistical test H0: the random distribution == the explanation distribution.
                    f_obs = []
                    for value in theoretical_distribution_lagged[:, 0]:
                        value = value / topKlagged
                        f_obs.append(precision_list_lagged.count(value))
                    f_exp = theoretical_distribution_lagged[:, 1] * len(precision_list_lagged)
                    _, p_value_lagged = scipy.stats.chisquare(f_obs, f_exp, ddof=0, axis=0)

                if topKnonlagged <= data_config['topKnonlagged_max']:
                    precision_mean_nonlagged = np.mean(precision_list_nonlagged)
                    recall_mean_nonlagged = np.mean(recall_list_nonlagged)
                    # precision_std_nonlagged = np.std(precision_list_nonlagged)
                    # recall_std_nonlagged = np.std(recall_list_nonlagged)
                    # precision_median_nonlagged = np.median(precision_list_nonlagged)
                    # recall_median_nonlagged = np.median(recall_list_nonlagged)
                    # precision_iqr_nonlagged = scipy.stats.iqr(precision_list_nonlagged)
                    # recall_iqr_nonlagged = scipy.stats.iqr(recall_list_nonlagged)

                    # statistical test : H0: the random distribution == the explanation distribution.
                    f_obs = []
                    for value in theoretical_distribution_nonlagged[:, 0]:
                        value = value / topKnonlagged
                        f_obs.append(precision_list_nonlagged.count(value))
                    f_exp = theoretical_distribution_nonlagged[:, 1] * len(precision_list_nonlagged)
                    _, p_value_nonlagged = scipy.stats.chisquare(f_obs, f_exp, ddof=0, axis=0)

                new_row = dict([("config_name", args.filename), ("dataset_name", data_config['dataset_name'])] +
                               [("filename", filename), ("target", TARGET_NAME)] +
                               [("model", data_config['modelname']), ("explanation", explanation_method)] +
                               [("topKlagged", topKlagged if topKlagged <= data_config['topKlagged_max'] else None)] +
                               [("topKnonlagged",
                                 topKnonlagged if topKnonlagged <= data_config['topKnonlagged_max'] else None)] +
                               [("precision@k_mean_lagged",
                                 precision_mean_lagged if topKlagged <= data_config['topKlagged_max'] else None)] +
                               [("precision@k_mean_nonlagged", precision_mean_nonlagged if topKnonlagged <= data_config[
                                   'topKnonlagged_max'] else None)] +
                               [("recall@k_mean_lagged",
                                 recall_mean_lagged if topKlagged <= data_config['topKlagged_max'] else None)] +
                               [("recall@k_mean_nonlagged", recall_mean_nonlagged if topKnonlagged <= data_config[
                                   'topKnonlagged_max'] else None)] +
                               # [("precision@k_std_lagged",precision_std_lagged)]+
                               # [("precision@k_std_nonlagged",precision_std_nonlagged)]+
                               # [("recall@k_std_lagged",recall_std_lagged)]+
                               # [("recall@k_std_nonlagged",recall_std_nonlagged)]+
                               # [("precision@k_median_lagged",precision_median_lagged)]+
                               # [("precision@k_median_nonlagged",precision_median_nonlagged)]+
                               # [("recall@k_median_lagged",recall_median_lagged)]+
                               # [("recall@k_median_nonlagged",recall_median_nonlagged)]+
                               # [("precision@k_iqr_lagged",precision_iqr_lagged)]+
                               # [("precision@k_iqr_nonlagged",precision_iqr_nonlagged)]+
                               # [("recall@k_iqr_lagged",recall_iqr_lagged)]+
                               # [("recall@k_iqr_nonlagged",recall_iqr_nonlagged)]+
                               [("random_precision_mean_nonlagged",
                                 random_precision_nonlagged if topKnonlagged <= data_config[
                                     'topKnonlagged_max'] else None)] +
                               [("random_precision_mean_lagged",
                                 random_precision_lagged if topKlagged <= data_config['topKlagged_max'] else None)] +
                               [("random_recall_mean_nonlagged",
                                 random_recall_nonlagged if topKnonlagged <= data_config[
                                     'topKnonlagged_max'] else None)] +
                               [("random_recall_mean_lagged",
                                 random_recall_lagged if topKlagged <= data_config['topKlagged_max'] else None)]
                               # [("random_pvalue_lagged",p_value_lagged)]+
                               # [("random_pvalue_nonlagged",p_value_nonlagged)]
                               )

                # add the more complicated case of consistency (lagged or not, with or without minority class)
                if data_config["minority_only"]:
                    new_row["consistency_lagged_minority"] = consistency_metric_lagged[
                        minority_class] if topKlagged <= data_config['topKlagged_max'] else None
                    new_row["consistency_nonlagged_minority"] = consistency_metric_nonlagged[
                        minority_class] if topKnonlagged <= data_config['topKnonlagged_max'] else None
                    new_row["consistency_lagged_random"] = \
                        np.log2(scipy.special.comb(len(VAR_NAMES) * pastPointsToForecast, topKlagged)) \
                        if topKlagged <= data_config['topKlagged_max'] else None
                    new_row["consistency_nonlagged_random"] = np.log2(scipy.special.comb(len(VAR_NAMES),
                                                                                         topKnonlagged))\
                        if topKnonlagged <= data_config['topKnonlagged_max'] else None

                else:
                    for i in range(data_config['numberLabels']):
                        if topKnonlagged > data_config['topKnonlagged_max']:
                            new_row["consistency_nonlagged_" + str(i)] = None
                        else:
                            new_row["consistency_nonlagged_" + str(i)] = consistency_metric_nonlagged[i]

                        if topKlagged > data_config['topKlagged_max']:
                            new_row["consistency_lagged_" + str(i)] = None
                        else:
                            new_row["consistency_lagged_" + str(i)] = consistency_metric_lagged[i]

                df_result_expl = pd.concat([df_result_expl, pd.DataFrame.from_records([new_row])])

                ################
                #
                #   Computing restricted model performances for predicted causes - NONLAGGED
                #
                ################

                if data_config['compute_nonlagged_restricted']:
                    if topKnonlagged <= data_config['topKnonlagged_max']:
                        # to make computation faster, only include for topK equal to the number of nonlagged parents
                        if topKnonlagged != len(NONLAGGED_CAUSAL_LIST):
                            continue

                        #   Find the top restricted features of equal size to true causes.

                        if data_config["minority_only"]:
                            freq = frequencies_nonlagged[minority_class]
                        else:
                            # normalize frequency for each label, add all labels togethers and select topK
                            sums = [sum(d.values()) for d in frequencies_nonlagged]
                            freq = defaultdict(lambda: 0)
                            for i, d in enumerate(frequencies_nonlagged):
                                for feature in d:
                                    freq[feature] += d[feature] / sums[i]

                        # verify all variables are in the set, and if not put their frequency to 0
                        for variable in VAR_NAMES:
                            if variable not in freq:
                                freq[variable] = 0

                        freq = list(freq.items())
                        restricted_features = np.argpartition(freq, -topKnonlagged, axis=0)[:, 1][-topKnonlagged:]
                        restricted_features = [freq[i][0] for i in restricted_features]

                        #   Restrict data to most frequently selected features

                        indexes = [VAR_NAMES.index(var) for var in restricted_features]
                        restricted_window_data = window_data[:, :, indexes]

                        #   Train

                        new_row = train_restricted_model(restricted_window_data, window_labels, fccv=True)
                        new_row["topKnonlagged"] = topKnonlagged
                        new_row["filename"] = filename
                        new_row["target"] = TARGET_NAME
                        new_row["explanation"] = explanation_method
                        new_row["lagged"] = False

                        df_result_restrict_classif_pred = pd.concat([df_result_restrict_classif_pred,
                                                                     pd.DataFrame.from_records([new_row])])

                ################
                #
                #   Computing restricted model performances for predicted causes - LAGGED
                #
                ################

                if data_config['compute_lagged_restricted']:
                    if topKlagged <= data_config['topKlagged_max']:
                        # to make computation faster, only include the result for topK equal to the number of causes
                        if topKlagged != len(LAGGED_CAUSAL_LIST):
                            continue

                        #   Restrict data to most frequently selected features

                        if data_config["minority_only"]:
                            freq = frequencies_lagged[minority_class]
                        else:
                            # normalize frequency for each label, add all labels togethers and select topK
                            sums = [sum(d.values()) for d in frequencies_lagged]
                            freq = defaultdict(lambda: 0)
                            for i, d in enumerate(frequencies_lagged):
                                for feature in d:
                                    freq[feature] += d[feature] / sums[i]

                        # verify all variables are in the set, and if not put their frequency to 0
                        for variable in VAR_NAMES:
                            for lag in range(1, data_config["pastPointsToForecast"]):
                                if "L" + str(lag) + "." + variable not in freq:
                                    freq["L" + str(lag) + "." + variable] = 0

                        freq = list(freq.items())
                        restricted_features = np.argpartition(freq, -topKlagged, axis=0)[:, 1][-topKlagged:]
                        restricted_features = set(freq[i][0] for i in restricted_features)

                        mask = np.array(
                            [["L" + str(lag) + "." + variable in restricted_features for variable in VAR_NAMES] for lag
                             in range(data_config["pastPointsToForecast"], 0, -1)])
                        restricted_window_data = window_data * mask.astype(int)

                        #   Train

                        new_row = train_restricted_model(restricted_window_data, window_labels, fccv=True)
                        new_row["topKlagged"] = topKlagged
                        new_row["filename"] = filename
                        new_row["target"] = TARGET_NAME
                        new_row["explanation"] = explanation_method
                        new_row["lagged"] = True

                        df_result_restrict_classif_pred = pd.concat([df_result_restrict_classif_pred,
                                                                     pd.DataFrame.from_records([new_row])])

        ################
        #
        #   Computing restricted model performances for true causes - NONLAGGED
        #
        ################

        if data_config["true_restricted"] and data_config['compute_nonlagged_restricted']:
            #   Restrict data to true parent causal variables

            restricted_features = NONLAGGED_CAUSAL_LIST
            indexes = [VAR_NAMES.index(var) for var in restricted_features]
            restricted_window_data = window_data[:, :, indexes]

            new_row = train_restricted_model(restricted_window_data, window_labels, fccv=True)
            new_row["lagged"] = False

            df_result_restrict_classif_true = pd.concat([df_result_restrict_classif_true,
                                                         pd.DataFrame.from_records([new_row])])

        ################
        #
        #   Computing restricted model performances for true causes - LAGGED
        #
        ################

        if data_config["true_restricted"] and data_config['compute_lagged_restricted']:
            mask = np.array(
                [["L" + str(lag) + "." + variable in LAGGED_CAUSAL_LIST for variable in VAR_NAMES] for lag in
                 range(data_config["pastPointsToForecast"], 0, -1)])
            restricted_window_data = window_data * mask.astype(int)

            #   Train

            new_row = train_restricted_model(restricted_window_data, window_labels, fccv=True)
            new_row["lagged"] = True
            new_row["filename"] = filename
            new_row["target"] = TARGET_NAME

            df_result_restrict_classif_true = pd.concat([df_result_restrict_classif_true,
                                                         pd.DataFrame.from_records([new_row])])

    ################
    #
    #   Save results - NON CONCURRENT.
    #
    ################

    if not df_result_classif.empty:
        df_result_classif_0 = pd.read_csv("./results/classif/" + args.filename + ".csv") if os.path.isfile(
            "./results/classif/" + args.filename + ".csv") else pd.DataFrame()
        df_result_classif = pd.concat([df_result_classif_0, df_result_classif], ignore_index=True, sort=False)
        df_result_classif.to_csv("./results/classif/" + args.filename + ".csv", index=False)

    if not df_result_restrict_classif_true.empty:
        df_result_restrict_classif_true_0 = pd.read_csv(
            "./results/restricted_classif_true/" + args.filename + ".csv") if os.path.isfile(
            "./results/restricted_classif_true/" + args.filename + ".csv") else pd.DataFrame()
        df_result_restrict_classif_true = pd.concat(
            [df_result_restrict_classif_true_0, df_result_restrict_classif_true], ignore_index=True, sort=False)
        df_result_restrict_classif_true.to_csv("./results/restricted_classif_true/" + args.filename + ".csv",
                                               index=False)

    if not df_result_restrict_classif_pred.empty:
        df_result_restrict_classif_pred_0 = pd.read_csv(
            "./results/restricted_classif_pred/" + args.filename + ".csv") if os.path.isfile(
            "./results/restricted_classif_pred/" + args.filename + ".csv") else pd.DataFrame()
        df_result_restrict_classif_pred = pd.concat(
            [df_result_restrict_classif_pred_0, df_result_restrict_classif_pred], ignore_index=True, sort=False)
        df_result_restrict_classif_pred.to_csv("./results/restricted_classif_pred/" + args.filename + ".csv",
                                               index=False)

    if not df_result_expl.empty:
        df_result_expl_0 = pd.read_csv("./results/expl_metrics/" + args.filename + ".csv") if os.path.isfile(
            "./results/expl_metrics/" + args.filename + ".csv") else pd.DataFrame()
        df_result_expl = pd.concat([df_result_expl_0, df_result_expl], ignore_index=True, sort=False)
        df_result_expl.to_csv("./results/expl_metrics/" + args.filename + ".csv", index=False)
