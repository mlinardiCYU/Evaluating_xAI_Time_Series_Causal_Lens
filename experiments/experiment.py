import os
import json
import argparse

import time
import pandas as pd
import numpy as np

from collections import defaultdict

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
#   Opening data
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
    #   Opening datasets: MTS + causal graph
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
    #   Creating the causal graphs in nx.Digraph form
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

        ################
        #
        #   If all explanations have been computed already, skip this target.
        #
        ################

        all_computed = []  # each element is true if the relevant file exists else false.
        for explanation_method in data_config["explanation_methods"]:
            savefile = "./saved_explanations/" + data_config["dir_to_save"] + os.path.splitext(filename)[
                0] + "_" + TARGET_NAME + "_" + explanation_method + ".npz"
            all_computed.append(os.path.isfile(savefile))

        if all(all_computed):
            continue

        ################
        #
        #   Separate into one or several folds
        #
        ################

        
        numberFolds = data_config['numberFolds']
        if numberFolds > 1:
            skf_object = sklearn.model_selection.StratifiedKFold(numberFolds, shuffle=True,
                                                                 random_state=np.random.RandomState(0))
            skf_generator = skf_object.split(window_data, window_labels)

        elif numberFolds == 1:
            # take the earlier 1-test_split_fraction percent of the data as training, and the latter as test.
            separation = int((1 - data_config['test_split_fraction']) * len(window_data))
            train_indices = list(range(separation))
            test_indices = list(range(separation, len(window_data)))
            skf_generator = [(train_indices, test_indices)]
            
            if np.count_nonzero(window_labels[:separation]==0) in [0,separation]:
                print("Non-stationary data for target variable: only one label in training data")
                continue
        else:
            raise Exception("numberFold argument to config file is invalid")
        
        ################
        #
        #   Train, predict, explain - loop
        #
        ################

        y_true = []
        y_pred = []
        y_score = []
        explanations = defaultdict(list)
        times = defaultdict(list)

        for ith_fold, (train_index, test_index) in enumerate(skf_generator):
            data_train = window_data[train_index]
            labels_train = window_labels[train_index]
            data_test = window_data[test_index]
            labels_test = window_labels[test_index]

            # oversampling
            new_index, labels_train = RandomOverSampler(random_state=legacy_rng).fit_resample(
                np.arange(len(data_train)).reshape((-1, 1)), labels_train)
            data_train = data_train[new_index.reshape((-1,))]

            # model declaration
            model_constructor_config = constructor_from_nbvar(len(VAR_NAMES))
            model = model_constructor(**model_constructor_config)

            # If we already computed a model with a previous run of a different explanation method (same config file)
            # Directly load the model
            save_path_model = "./saved_models/" + data_config["dir_to_save"] + os.path.splitext(filename)[
                0] + "_" + TARGET_NAME + "_" + str(ith_fold)
            if os.path.exists(save_path_model) and data_config["load_model"]:
                model.load(save_path_model)
            else:
                model.train(data_train, labels_train)
                # save if necessary
                if data_config["load_model"]:
                    model.save(save_path_model)

            # obtain predicted labels from the test set
            score_pred, labels_pred = model.predict(data_test)

            # compute the explanations
            for explanation_method in data_config["explanation_methods"]:
                # verify if explanation was computed already
                savefile = "./saved_explanations/" + data_config["dir_to_save"] + os.path.splitext(filename)[
                    0] + "_" + TARGET_NAME + "_" + explanation_method + ".npz"
                if os.path.isfile(savefile):
                    continue

                # compute explanations
                INITIAL_TIME = time.time()
                if explanation_method in ["IG", "GS", "FA", "FP", "OS", "DM"]:
                    explanation = model.explain(data_test, labels_test, labels_pred, explanation_method)
                    explanations[explanation_method].extend(explanation)
                else:
                    for i in range(len(data_test)):
                        data_sample = data_test[i]
                        true_label_sample = labels_test[i]
                        pred_label_sample = labels_pred[i]

                        explanation = model.explain(data_sample, true_label_sample, pred_label_sample,
                                                    explanation_method)
                        # keep track of results
                        explanations[explanation_method].append(explanation)

                FINAL_TIME = time.time() - INITIAL_TIME
                times[explanation_method].append(FINAL_TIME / len(data_test))

            # keep track of results
            y_pred.append(labels_pred)
            y_true.append(labels_test)
            y_score.append(score_pred)

        # convert to array
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)
        mean_times = dict()

        # average time over folds
        for explanation_method in data_config["explanation_methods"]:
            mean_times[explanation_method] = np.array(np.mean(times[explanation_method]))

        ################
        #
        #   Removing explanations that failed (None values) (that might happen for dCAM)
        #
        ################

        for explanation_method in data_config["explanation_methods"]:

            none_index = [i for i, x in enumerate(explanations[explanation_method]) if x is not None]

            if none_index == []:
                print("\t\tNo explanation found: (file,target,explanation) not taken into account")
                continue
            y_true_temp, y_pred_temp, y_score_temp = y_true[none_index], y_pred[none_index], y_score[none_index]
            explanations[explanation_method] = [explanations[explanation_method][i] for i in none_index]
            expl = np.array(explanations[explanation_method])

            ################
            #
            #   Saving/opening results to be able to open them quickly if code changes
            #
            ################

            savefile = "./saved_explanations/" + data_config["dir_to_save"] + os.path.splitext(filename)[
                0] + "_" + TARGET_NAME + "_" + explanation_method + ".npz"
            np.savez(savefile, y_true=y_true_temp, y_pred=y_pred_temp, y_score=y_score_temp, explanations=expl,
                     times=mean_times[explanation_method])

