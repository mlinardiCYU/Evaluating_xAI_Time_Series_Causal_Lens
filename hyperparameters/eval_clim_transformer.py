from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import nn
import torch
import sys
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import optuna

import scipy.stats
import scipy.special
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import RandomOverSampler

from collections import defaultdict


rng = np.random.default_rng(0)  # random seed
legacy_rng = np.random.RandomState(0)

# ADD ALL NECESSARY LIBRARIES FOR THE EXPERIMENT
sys.path.insert(0, '../models')
from transformer.Transformer import Transformer, CosineWarmupScheduler


PAST_POINTS_TO_FORECAST = 8


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class model_constructor():

    def __init__(self, input_dim, seq_len, model_dim, num_classes, num_heads, num_layers, lr, batch_size, warmup, num_epochs, device, dropout=0.0):
        # input parameters
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_classes = num_classes

        # training parameters
        self.lr = lr
        self.batch_size = batch_size
        self.warmup = warmup
        self.device = device
        self.num_epochs = num_epochs

        self.model = Transformer(input_dim, seq_len, model_dim,
                                 num_classes, num_heads, num_layers, dropout).to(self.device)

    def _prepare_data_train(self, data, labels, shuffle=True, batch_size=32):
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data = data.reshape(data.shape[0], self.seq_len, self.input_dim)
        dataset = data_utils.TensorDataset(
            torch.from_numpy(data), torch.from_numpy(labels))
        loader = data_utils.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def _prepare_data_test(self, data):
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        data = data.reshape(data.shape[0], self.seq_len, self.input_dim)
        dataset = data_utils.TensorDataset(
            torch.from_numpy(data))
        loader = data_utils.DataLoader(
            dataset, batch_size=512, shuffle=False)
        return loader

    def train(self, data_train, labels_train):
        train_loader = self._prepare_data_train(data_train, labels_train)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0001)
        scheduler = CosineWarmupScheduler(
            optimizer, self.warmup, self.num_epochs)  # max_iters = self.num_epochs

        progress_bar = tqdm(
            range(self.num_epochs),
            total=self.num_epochs,
        )

        self.model.train()

        for epoch in progress_bar:

            for step, (samples, labels) in enumerate(train_loader):
                # Make sure samples are in the right format
                samples = samples.reshape(-1, self.seq_len,
                                          self.input_dim).to(self.device)
                # Cast and enable autograd
                samples = Variable(samples).float().to(self.device)
                # cast to long as per nn.CrossEntropyLoss specification
                labels = Variable(labels).long().to(self.device)

                optimizer.zero_grad()

                # Compute outputs and loss
                outputs = self.model(samples).to(self.device)

                loss = loss_fn(outputs, labels)
                # Retropragate gradients
                loss.backward()
                # Update parameters
                optimizer.step()
                scheduler.step()

                if step % 10 == 9:
                    lr = optimizer.param_groups[0]["lr"]
                    progress_bar.set_postfix(
                        lr=f"{lr:.1E}", loss=f"{loss.item():.3f}")

        progress_bar.close()

    def predict(self, data_test):
        test_loader = self._prepare_data_test(data_test)
        y_pred = []
        y_score = []
        with torch.no_grad():
            self.model.eval()
            for batch in test_loader:
                samples = batch[0]
                samples = samples.reshape(-1, self.seq_len,
                                          self.input_dim).to(self.device)

                samples = Variable(samples).float().to(self.device)
                outputs = self.model(samples).to(self.device)
                # _, predicted = torch.max(outputs.data, 1)
                non_normalized_out = outputs.data.detach().cpu().numpy()
                y_pred.append(np.argmax(non_normalized_out, axis=-1))
                if self.num_classes == 2:
                    normalized_proba = scipy.special.softmax(
                        non_normalized_out, axis=1)[:, 1]
                    y_score.append(normalized_proba)

        if self.num_classes == 2:
            return np.concatenate(y_score), np.concatenate(y_pred)

    def explain(self, data_sample, true_label, pred_label, method):
        raise NotImplementedError

    def summary(self):
        print(self.model)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)


def get_final_data(dataset_name, filename):

    df = pd.read_csv("../data/"+dataset_name+"/" +
                     filename, header=None, sep=" ")
    df.columns = [str(i) for i in df.columns]

    # df=df[df.columns[1:]]
    VAR_NAMES = list(df.columns)

    pastPointsToForecast = 8
    numberFolds = 3

    # prepare data

    final_data = defaultdict(list)

    for TARGET_NAME in VAR_NAMES:
        data_for_labels = df[TARGET_NAME].values.reshape((-1, 1))

        est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')
        est.fit(data_for_labels)
        labels = est.transform(data_for_labels)
        labels = labels.reshape((-1,))

        window_data = []
        window_labels = []

        for windowbegin in range(df.values.shape[0]-pastPointsToForecast):
            current_window = df.values[windowbegin:windowbegin +
                                       pastPointsToForecast, :]
            current_label = labels[windowbegin+pastPointsToForecast]
            window_data.append(current_window)
            window_labels.append(current_label)

        window_data = np.array(window_data)
        window_labels = np.array(window_labels)

        # EXCLUDE THE LATEST 30% FRAMES THAT MAKE THE TEST SET OF THE EXPERIMENTS
        separation = int((1-0.3)*len(window_data))
        window_data = window_data[:separation]
        window_labels = window_labels[:separation]

        skf_object = sklearn.model_selection.StratifiedKFold(
            numberFolds, shuffle=True, random_state=legacy_rng)
        skf_generator = skf_object.split(window_data, window_labels)

        for ith_fold, (train_index, test_index) in enumerate(skf_generator):
            data_train = window_data[train_index]
            labels_train = window_labels[train_index]
            data_test = window_data[test_index]
            labels_test = window_labels[test_index]

            # insert oversampling here
            new_index, labels_train = RandomOverSampler(random_state=legacy_rng).fit_resample(
                np.arange(len(data_train)).reshape((-1, 1)), labels_train)
            data_train = data_train[new_index.reshape((-1,))]

            final_data[TARGET_NAME].append(
                [data_train, labels_train, data_test, labels_test])

    return final_data, VAR_NAMES


def objective(trial, final_data, var_names):

    TRIAL_number_of_layers = trial.suggest_int("num_layers", 1, 7)
    TRIAL_model_dim = trial.suggest_categorical("model_dim", [16, 32, 64, 128, 256])
    TRIAL_number_of_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])
    TRIAL_num_epochs = trial.suggest_int("num_epochs", 10, 150, 15)
    TRIAL_lr = trial.suggest_float("lr", 0.00001, 0.001)

    optimize_score = 0
    for TARGET_NAME in var_names:

        input_dim = len(var_names)
        seq_len = PAST_POINTS_TO_FORECAST
        num_classes = 2

        num_heads = TRIAL_number_of_heads
        num_layers = TRIAL_number_of_layers
        model_dim = TRIAL_model_dim

        num_epochs = TRIAL_num_epochs
        lr = TRIAL_lr
        batch_size = 32
        warmup = num_epochs//5
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dropout = 0.1

        config = {"device": device,
                  "input_dim": input_dim,
                  "seq_len": seq_len,
                  "model_dim": model_dim,
                  "num_classes": num_classes,
                  "num_heads": num_heads,
                  "num_layers": num_layers,
                  "lr": lr,
                  "batch_size": batch_size,
                  "num_epochs": TRIAL_num_epochs,
                  "warmup": warmup,
                  "dropout": dropout}

        y_true = []
        y_score = []
        y_pred = []

        for data_train, labels_train, data_test, labels_test in final_data[TARGET_NAME]:

            # insert model declaration here
            print(config)
            model = model_constructor(**config)
            # print(model.summary())

            # insert model training here
            model.train(data_train, labels_train)  # replace as necessary

            # obtain predicted labels from the test set
            score_pred, labels_pred = model.predict(
                data_test)  # replace as necessary

            y_score.append(score_pred)
            y_pred.append(labels_pred)
            y_true.append(labels_test)

        # convert to array
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        #balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true=y_true,y_pred=y_pred)
        roc_auc_score = sklearn.metrics.roc_auc_score(y_true, y_score)

        optimize_score += roc_auc_score

    return -optimize_score


def main():

    dataset_name = "TestCLIM_N-5_T-250/returns"

    filenames = [
                "TestCLIM_N-5_T-250_0035.txt",
                "TestCLIM_N-5_T-250_0002.txt",
                "TestCLIM_N-5_T-250_0139.txt"
                ]

    for filename in filenames:
        final_data, var_names = get_final_data(dataset_name=dataset_name, filename=filename)
        n_trials = 100
        study = optuna.create_study()
        begin_time = time.time()
        study.optimize(lambda trial: objective(trial, final_data, var_names),n_trials=n_trials)
        print("Time spent", time.time()-begin_time)
        print(study.best_params)
        print(study.best_value)

        with open("results_transformer_clim.txt", "a") as f:
            s = "\ntransformer model, "+dataset_name+"/"+filename
            s = s+"\nNumber of trials:"+str(n_trials)
            s = s+"\nTime spent:"+str(int(time.time()-begin_time))
            s = s+"\n"+str(study.best_params)+"\n" + \
                "optimized_goal: "+str(study.best_value)
            s = s+"\n\n"
            f.write(s)


if __name__ == "__main__":
    main()
