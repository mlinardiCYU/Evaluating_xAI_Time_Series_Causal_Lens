import sys

import numpy as np
import scipy.special

import keras.optimizers

from torchsummary import summary
import torch
from torch import nn

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    FeaturePermutation,
    FeatureAblation,
    Occlusion
)

from tqdm import tqdm
from torch.autograd import Variable
import torch.utils.data as data_utils

rootdir = "../"
sys.path.insert(0, rootdir + '/models')

from XCM_pytorch.models.xcm_pytorch import XCM_pyTorch, ModelCNN, TSDataset

from transformer.Transformer import Transformer, CosineWarmupScheduler

sys.path.insert(0, rootdir + '/models/dCAM/src/models')
sys.path.insert(0, rootdir + '/models/dCAM/src/explanation')

import CNN_models
from DCAM import *


sys.path.insert(0, rootdir + '/models/LSTM_TSIB/')

import LSTM

# Dynamask imports
sys.path.insert(0, rootdir + '/dynamask')
from mask_group import MaskGroup
from perturbation import GaussianBlur
from losses import mse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###########
#
# Model wrapper for XCM_pytorch
#
###########


class XCM_model_constructor:

    def _prepare_training_data(self, data, labels, batchsize):
        dataset = CNN_models.TSDataset(np.array(data), np.array(labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize)
        return dataloader


    def __init__(self, device, input_shape, n_class, window_size, filters_num, batch_size, epochs):

        self.input_shape = input_shape
        self.n_class = n_class
        self.window_size = window_size
        self.filters_num = filters_num
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        xcm_NET = XCM_pyTorch(n=input_shape[0], k=input_shape[1], window_size=window_size, n_class=n_class)
        self.model = ModelCNN(xcm_NET, device=self.device)
        
        

    def train(self, data_train, labels_train):
        #create correct dataloader
        dataloader_training = self._prepare_training_data(data_train,labels_train, self.batch_size)
        #train the model
        self.model.train(self.epochs, dataloader_training, model_name='XCM')


    def predict(self, data_test):
        return self.model.predict(data_test,self.n_class)

    def explain(self, data_sample, true_label, pred_label, method):
        self.model.model.eval()
        if method == "XCM-2D":
            data_sampleArray = np.reshape(data_sample,(data_sample.shape[0],data_sample.shape[1],1))
            expl = self.model.grad_cam_Pytorch(data_sampleArray, "2D", label = true_label)
            expl = np.swapaxes(expl, -1, -2)  # explanations should be in the form (D,T) by code convention
            return expl
        elif method == "XCM-1D":
            data_sampleArray = np.reshape(data_sample,(data_sample.shape[0],data_sample.shape[1],1))
            expl = self.model.grad_cam_Pytorch(data_sampleArray, "1D", label = true_label, produce_plot=True)
            expl = np.swapaxes(expl, -1, -2)  # explanations should be in the form (D,T) by code convention
            return expl
        else:
        
            self.explained_model = self.model.model
            explained_model = self.explained_model
            explained_model.eval()

            loader = CNN_models.TSDataset(data_sample, pred_label)
            loader = torch.utils.data.DataLoader(loader, batch_size=self.batch_size, shuffle=False)

            explanations = []
            # Model agnostic explainability implementations accept input batches
            for sample, label in loader:
                inputs = sample
                inputs = Variable(inputs, volatile=False,
                                  requires_grad=True).float().to(self.device)
                label = label.type(torch.int64).to(self.device)

                # Random baselines for IntegratedGradients, GradientShape and Occlusion
                baseline_single = torch.from_numpy(
                    np.random.random(inputs.shape)).float().to(self.device)

                baseline_multiple = torch.from_numpy(np.random.random(
                    (inputs.shape[0] * 5, inputs.shape[1], inputs.shape[2]))).float().to(self.device)

                # Permutation feature mask
                mask = np.zeros(
                    (self.input_shape[0], self.input_shape[1]), dtype=int)
                for i in range(self.input_shape[0]):
                    mask[i, :] = i

                mask_single = torch.from_numpy(mask).to(self.device)
                mask_single = mask_single.reshape(1, self.input_shape[0], self.input_shape[1]).int().to(self.device)

                if method == "IG":
                    IG = IntegratedGradients(explained_model)
                    e = IG.attribute(inputs, baselines=baseline_single,
                                     target=label).detach().cpu().numpy()

                elif method == "GS":
                    GS = GradientShap(explained_model)
                    e = GS.attribute(inputs, baselines=baseline_multiple,
                                     stdevs=0.09,
                                     target=label).detach().cpu().numpy()

                elif method == "FP":
                    FP = FeaturePermutation(explained_model)
                    e = FP.attribute(inputs, target=label,
                                     perturbations_per_eval=inputs.shape[0]).detach().cpu().numpy()

                elif method == "FA":
                    FA = FeatureAblation(explained_model)
                    e = FA.attribute(
                        inputs, target=label).detach().cpu().numpy()

                elif method == "OS":
                    # the method computes the change when a rectangle is hidden.
                    # the feature score is the average change of all rectangle that includes it.
                    # for the current experiment, the score per rectangle can be traced back by the cumulative diff
                    # starting from the end of the sequence.
                    OS = Occlusion(explained_model)
                    e = OS.attribute(inputs,
                                     sliding_window_shapes=(2, self.input_shape[1]),
                                     target=label,
                                     baselines=baseline_single).detach().cpu().numpy()
                elif method == "DM":
                    explained_model.to(self.device)

                    e = []
                    areas = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
                    for input in tqdm(inputs):
                        pert = GaussianBlur(self.device)
                        mask = MaskGroup(pert, self.device, verbose=False)
                        mask.fit(input, explained_model, loss_function=mse, area_list=areas, size_reg_factor_init=0.01, n_epoch=100)
                        sample_e = mask.masks_tensor.detach().cpu().numpy()
                        e.append(sample_e)
                else:
                    raise Exception("Provided explanation method is not implemented in the model wrapper class.")

                explanations.extend(e)

            explanations = np.array(explanations)
            # the code standard is to get explanations of the form (D,T), but here we get (T,D).
            explanations = np.swapaxes(explanations, -1, -2)


            return explanations

    def summary(self):
        return summary(self.model.model.to(self.device), input_size=tuple(self.input_shape))

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)


###########
#
# Copy of model construction of dCAM as some hyperparameters will be changed
#
###########


class ConvNet2D(nn.Module):
    def __init__(self, original_length, original_dim, nb_channel, num_classes=10, TRIAL_number_of_layers=6,
                 TRIAL_number_of_channels=128, TRIAL_time_kernel=3):
        super(ConvNet2D, self).__init__()

        self.TRIAL_number_of_layers = TRIAL_number_of_layers
        self.TRIAL_number_of_channels = TRIAL_number_of_channels
        self.TRIAL_time_kernel = TRIAL_time_kernel
        self.kernel_size = (1, self.TRIAL_time_kernel)
        self.padding = "same"
        self.num_class = num_classes

        if self.TRIAL_number_of_layers == 1:
            self.layer1 = nn.Sequential(
                nn.Conv2d(nb_channel, TRIAL_number_of_channels, kernel_size=self.kernel_size, padding=self.padding),
                nn.ReLU(),
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(nb_channel, TRIAL_number_of_channels, kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels),
                nn.ReLU(),
            )

        if self.TRIAL_number_of_layers == 2:
            self.layer2 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.ReLU(),
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels),
                nn.ReLU(),
            )

        if self.TRIAL_number_of_layers == 3:
            self.layer21 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels * 2, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.ReLU(),
            )
        else:
            self.layer21 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels * 2, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels * 2),
                nn.ReLU(),
            )

        if self.TRIAL_number_of_layers == 4:
            self.layer22 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels * 2, TRIAL_number_of_channels * 2, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.ReLU(),
            )
        else:
            self.layer22 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels * 2, TRIAL_number_of_channels * 2, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels * 2),
                nn.ReLU(),
            )

        if self.TRIAL_number_of_layers == 5:
            self.layer23 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels * 2, TRIAL_number_of_channels * 2, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.ReLU(),
            )
        else:
            self.layer23 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels * 2, TRIAL_number_of_channels * 2, kernel_size=self.kernel_size,
                          padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels * 2),
                nn.ReLU(),
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(TRIAL_number_of_channels * 2, TRIAL_number_of_channels * 2, kernel_size=self.kernel_size,
                      padding=self.padding),
            nn.ReLU(),
        )

        self.GAP = nn.AvgPool2d(kernel_size=(original_dim, original_length))

        outshape = [TRIAL_number_of_channels,
                    TRIAL_number_of_channels,
                    TRIAL_number_of_channels * 2,
                    TRIAL_number_of_channels * 2,
                    TRIAL_number_of_channels * 2,
                    TRIAL_number_of_channels * 2][self.TRIAL_number_of_layers - 1]
        self.fc1 = nn.Sequential(nn.Linear(outshape, num_classes))

    def forward(self, x):
        if self.TRIAL_number_of_layers > 0:
            out = self.layer1(x)
        if self.TRIAL_number_of_layers > 1:
            out = self.layer2(out)
        if self.TRIAL_number_of_layers > 2:
            out = self.layer21(out)
        if self.TRIAL_number_of_layers > 3:
            out = self.layer22(out)
        if self.TRIAL_number_of_layers > 4:
            out = self.layer23(out)
        if self.TRIAL_number_of_layers > 5:
            out = self.layer3(out)

        out = self.GAP(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out


###########
#
# Model that combine preprocessing and dCAM for captum explainability algorithms.
#
###########

class dCAM_captum_model(torch.nn.Module):
    def __init__(self, main_model, original_length, original_dim, device):
        super(dCAM_captum_model, self).__init__()
        self.main_model = main_model
        self.original_length = original_length
        self.original_dim = original_dim
        self.device = device

    def forward(self, x):
        if len(x.shape) == 2 : 
            x = torch.unsqueeze(x, dim=0) # single sample support (for DM) (T, D) => (1, T, D)
        out = torch.transpose(x, 1, 2).to(self.device)  # swap time and dim
        out = torch.unsqueeze(out, dim=1)  # add new dimension
        out = torch.tile(out, (1, self.original_dim, 1, 1)).to(self.device)  # replicate on new dimension

        A = torch.unsqueeze(torch.unsqueeze(torch.arange(self.original_dim), dim=-1), dim=-1)
        A = torch.tile(A, (x.size()[0], 1, self.original_dim, self.original_length))
        B = torch.unsqueeze(torch.arange(self.original_dim), dim=-1)
        B = torch.tile(B, (x.size()[0], self.original_dim, 1, self.original_length))
        gatherer = (A + B) % self.original_dim  # indices by witch to recombine (rotate)
        out = torch.gather(out, dim=2, index=gatherer.to(self.device))

        out = self.main_model(out)
        return out


###########
#
# Model wrapper declaration for dCAM
#
###########


class dCAM_model_constructor:
    def gen_cube(self, instance):
        # rotate the features index to obtain the right format
        instance = instance.T
        result = []
        for i in range(len(instance)):
            result.append([instance[(i + j) % len(instance)] for j in range(len(instance))])
        return np.array(result)

    def _prepare_data_train(self, data, labels, shuffle=True):
        all_class = np.array([self.gen_cube(acl) for acl in data])
        dataset_mat = CNN_models.TSDataset(all_class, labels)
        loader = torch.utils.data.DataLoader(dataset_mat, batch_size=self.batch_size, shuffle=shuffle)
        return loader

    def _prepare_data_test(self, data):
        all_class = np.array([self.gen_cube(acl) for acl in data])
        dataset_mat = CNN_models.TSDataset(all_class, np.zeros((all_class.shape[0],)))
        loader = torch.utils.data.DataLoader(dataset_mat, batch_size=self.batch_size, shuffle=False)
        return loader

    def __init__(self, device,
                 original_length,
                 original_dim, num_classes,
                 batch_size, num_epochs,
                 nb_permutations,
                 number_of_layers,
                 number_of_channels,
                 time_kernel):

        self.device = device
        self.original_length = original_length
        self.original_dim = original_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.nb_permutations = nb_permutations
        self.number_of_layers = number_of_layers
        self.number_of_channels = number_of_channels
        self.time_kernel = time_kernel

        modelarch = ConvNet2D(self.original_length, self.original_dim, self.original_dim, self.num_classes,
                              TRIAL_number_of_layers=self.number_of_layers,
                              TRIAL_number_of_channels=self.number_of_channels, TRIAL_time_kernel=self.time_kernel).to(
            self.device)
        self.model = CNN_models.ModelCNN(modelarch, self.device, verbose=False)

        self.explained_model_built = False
        self.explained_model = None

    def train(self, data_train, labels_train):
        loader = self._prepare_data_train(data_train, labels_train, shuffle=True)
        self.model.train(self.num_epochs, dataloader_cl1=loader, dataloader_cl1_test=[])

    def predict(self, data_test):
        loader = self._prepare_data_test(data_test)

        y_score = []
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                self.model.model.eval()
                batch, _ = batch
                batch = torch.autograd.Variable(batch.float()).to(self.device)
                output = self.model.model(batch.float()).to(self.device)
                non_normalized_out = output.data.detach().cpu().numpy()
                y_pred.append(np.argmax(non_normalized_out, axis=-1))
                if self.num_classes == 2:
                    normalized_proba = scipy.special.softmax(non_normalized_out, axis=1)[:, 1]
                    y_score.append(normalized_proba)

        if self.num_classes == 2:
            return np.concatenate(y_score), np.concatenate(y_pred)

    def explain(self, data_sample, true_label, pred_label, method):
        self.model.model.eval()
        if method == "dCAM":
            data_sample = data_sample.T
            model_expl = self.model.model.to(self.device)
            last_conv_layer = model_expl._modules['layer3']
            fc_layer_name = model_expl._modules['fc1']

            DCAM_m = DCAM(model_expl, device, last_conv_layer=last_conv_layer, fc_layer_name=fc_layer_name)
            try:
                dcam, permutation_success = DCAM_m.run(
                    instance=data_sample,
                    nb_permutation=self.nb_permutations,
                    label_instance=pred_label)
            except:
                return None
            return dcam
        else:
            if not self.explained_model_built:
                # we cannot use original model as the input is of shape (batch,D,D,T): explanations would be (D,D,T) too
                self.explained_model = dCAM_captum_model(self.model.model, self.original_length, self.original_dim,
                                                         self.device)
                self.explained_model_built = True
            explained_model = self.explained_model
            explained_model.eval()

            loader = CNN_models.TSDataset(data_sample, pred_label)
            loader = torch.utils.data.DataLoader(loader, batch_size=self.batch_size, shuffle=False)

            explanations = []
            # Model agnostic explainability implementations accept input batches
            for sample, label in loader:
                inputs = sample
                inputs = Variable(inputs, volatile=False,
                                  requires_grad=True).float().to(device)
                label = label.type(torch.int64).to(device)

                # Random baselines for IntegratedGradients, GradientShape and Occlusion
                baseline_single = torch.from_numpy(
                    np.random.random(inputs.shape)).float().to(device)

                baseline_multiple = torch.from_numpy(np.random.random(
                    (inputs.shape[0] * 5, inputs.shape[1], inputs.shape[2]))).float().to(device)

                # Permutation feature mask
                mask = np.zeros(
                    (self.original_length, self.original_dim), dtype=int)
                for i in range(self.original_length):
                    mask[i, :] = i

                mask_single = torch.from_numpy(mask).to(device)
                mask_single = mask_single.reshape(1, self.original_length, self.original_dim).int().to(device)

                if method == "IG":
                    IG = IntegratedGradients(explained_model)
                    e = IG.attribute(inputs, baselines=baseline_single,
                                     target=label).detach().cpu().numpy()

                elif method == "GS":
                    GS = GradientShap(explained_model)
                    e = GS.attribute(inputs, baselines=baseline_multiple,
                                     stdevs=0.09,
                                     target=label).detach().cpu().numpy()

                elif method == "FP":
                    FP = FeaturePermutation(explained_model)
                    e = FP.attribute(inputs, target=label,
                                     perturbations_per_eval=inputs.shape[0]).detach().cpu().numpy()

                elif method == "FA":
                    FA = FeatureAblation(explained_model)
                    e = FA.attribute(
                        inputs, target=label).detach().cpu().numpy()

                elif method == "OS":
                    OS = Occlusion(explained_model)
                    e = OS.attribute(inputs,
                                     sliding_window_shapes=(2, self.original_dim),
                                     target=label,
                                     baselines=baseline_single).detach().cpu().numpy()
                elif method == "DM":
                    #model = explained_model.to(self.device)

                    e = []
                    areas = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
                    for input in tqdm(inputs):
                        pert = GaussianBlur(self.device)
                        mask = MaskGroup(pert, self.device, verbose=False)
                        mask.fit(input, explained_model, loss_function=mse, area_list=areas, size_reg_factor_init=0.01, n_epoch=100)
                        sample_e = mask.masks_tensor.detach().cpu().numpy()
                        e.append(sample_e)
                else:
                    raise Exception("Provided explanation method is not implemented in the model wrapper class.")

                explanations.extend(e)

            explanations = np.array(explanations)
            # the code standard is to get explanations of the form (D,T), but here we get (T,D).
            explanations = np.swapaxes(explanations, -1, -2)
            return explanations

    def summary(self):
        return summary(self.model.model.to(self.device),
                       input_size=(self.original_dim, self.original_dim, self.original_length))

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)


class LSTM_model_constructor:

    def __init__(self, device, epochs, batch_size, input_size, hidden_size, num_classes, rnndropout, original_length):
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnndropout = rnndropout
        self.epochs = epochs
        self.batch_size = batch_size

        self.original_dim = self.input_size
        self.original_length = original_length

        self.model = LSTM.LSTM(self.device, self.input_size, self.hidden_size, self.num_classes, self.rnndropout)
        self.explained_model = None

    def train(self, data_train, labels_train):
        data_train = CNN_models.TSDataset(data_train, labels_train)
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        self.model.to(self.device)
        self.model.train()
        for epochs in range(self.epochs):
            for i, (samples, labels) in enumerate(train_loader):
                samples = torch.autograd.Variable(samples.float()).to(self.device)
                labels = torch.autograd.Variable(labels.float()).type(torch.LongTensor).to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # predict classes
                outputs = self.model(samples)
                # compute the loss based on model output and real labels
                loss = loss_fn(outputs, labels)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                optimizer.step()

    def predict(self, data_test):
        loader = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, shuffle=False)

        y_score = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:

                batch = torch.autograd.Variable(batch.float()).to(self.device)
                output = self.model(batch.float()).to(self.device)
                non_normalized_out = output.data.detach().cpu().numpy()
                y_pred.append(np.argmax(non_normalized_out, axis=-1))
                if self.num_classes == 2:
                    normalized_proba = scipy.special.softmax(non_normalized_out, axis=1)[:, 1]
                    y_score.append(normalized_proba)

        if self.num_classes == 2:
            return np.concatenate(y_score), np.concatenate(y_pred)

    def explain(self, data_sample, true_label, pred_label, method):
        # we cannot use original model as the input is of shape (batch,D,D,T): explanations would be (D,D,T) too.
        self.explained_model = self.model
        explained_model = self.explained_model
        explained_model.eval()

        # seems like cudnn does not support gradient computation in eval mode for RNN...
        torch.backends.cudnn.enabled = False

        loader = CNN_models.TSDataset(data_sample, pred_label)
        loader = torch.utils.data.DataLoader(loader, batch_size=self.batch_size, shuffle=False)

        explanations = []
        # Model agnostic explainability implementations accept input batches
        for sample, label in loader:
            inputs = sample
            inputs = Variable(inputs, volatile=False,
                              requires_grad=True).float().to(self.device)
            label = label.type(torch.int64).to(self.device)

            # Random baselines for IntegratedGradients, GradientShape and Occlusion
            baseline_single = torch.from_numpy(
                np.random.random(inputs.shape)).float().to(self.device)

            baseline_multiple = torch.from_numpy(np.random.random(
                (inputs.shape[0] * 5, inputs.shape[1], inputs.shape[2]))).float().to(self.device)

            # Permutation feature mask
            mask = np.zeros(
                (self.original_length, self.original_dim), dtype=int)
            for i in range(self.original_length):
                mask[i, :] = i

            mask_single = torch.from_numpy(mask).to(self.device)
            mask_single = mask_single.reshape(1, self.original_length, self.original_dim).int().to(self.device)

            if method == "IG":
                IG = IntegratedGradients(explained_model)
                e = IG.attribute(inputs, baselines=baseline_single,
                                 target=label).detach().cpu().numpy()

            elif method == "GS":
                GS = GradientShap(explained_model)
                e = GS.attribute(inputs, baselines=baseline_multiple,
                                 stdevs=0.09,
                                 target=label).detach().cpu().numpy()

            elif method == "FP":
                FP = FeaturePermutation(explained_model)
                e = FP.attribute(inputs, target=label,
                                 perturbations_per_eval=inputs.shape[0]).detach().cpu().numpy()

            elif method == "FA":
                FA = FeatureAblation(explained_model)
                e = FA.attribute(
                    inputs, target=label).detach().cpu().numpy()

            elif method == "OS":
                # the method computes the change when a rectangle is hidden.
                # the feature score is the average change of all rectangle that includes it.
                # for the current experiment, the score per rectangle can be traced back by the cumulative diff
                # starting from the end of the sequence.
                OS = Occlusion(explained_model)
                e = OS.attribute(inputs,
                                 sliding_window_shapes=(2, self.original_dim),
                                 target=label,
                                 baselines=baseline_single).detach().cpu().numpy()
            elif method == "DM":
                #model = explained_model.to(self.device)

                e = []
                areas = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
                for input in tqdm(inputs):
                    pert = GaussianBlur(self.device)
                    mask = MaskGroup(pert, self.device, verbose=False)
                    mask.fit(input, explained_model, loss_function=mse, area_list=areas, size_reg_factor_init=0.01, n_epoch=100)
                    sample_e = mask.masks_tensor.detach().cpu().numpy()
                    e.append(sample_e)
            else:
                raise Exception("Provided explanation method is not implemented in the model wrapper class.")

            explanations.extend(e)

        explanations = np.array(explanations)
        # the code standard is to get explanations of the form (D,T), but here we get (T,D).
        explanations = np.swapaxes(explanations, -1, -2)

        torch.backends.cudnn.enabled = True  # setting back cudnn

        return explanations

    def summary(self):
        return summary(self.model.to(self.device), input_size=(self.original_length, self.original_dim))

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)


###########
#
# Utility class for Transformer
#
###########


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
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


###########
#
# Model wrapper declaration for Transformer
#
###########


class Transformer_model_constructor:

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

        self.model = Transformer(input_dim, seq_len, model_dim, num_classes, num_heads, num_layers, dropout).to(self.device)
        

    
    def _prepare_data_train(self, data, labels, shuffle=True, batch_size=32):
        data = data.reshape(data.shape[0], self.seq_len, self.input_dim)
        dataset = data_utils.TensorDataset(
            torch.from_numpy(data), torch.from_numpy(labels))
        loader = data_utils.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def _prepare_data_test(self, data):
        data = data.reshape(data.shape[0], self.seq_len, self.input_dim)
        dataset = data_utils.TensorDataset(
            torch.from_numpy(data))
        loader = data_utils.DataLoader(
            dataset, batch_size=512, shuffle=False)
        return loader

    def train(self, data_train, labels_train):
        train_loader = self._prepare_data_train(data_train, labels_train)

        loss_fn = nn.CrossEntropyLoss()
        optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0001)       
        scheduler = CosineWarmupScheduler(optimizer, self.warmup, self.num_epochs) # max_iters = self.num_epochs


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
                labels = Variable(labels).long().to(self.device) # cast to long as per nn.CrossEntropyLoss specification

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
                                          self.input_dim).to(device)
                outputs = self.model(samples)
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
        loader = self._prepare_data_train(
            data_sample, true_label, shuffle=False)
        explanations = []

        for sample, labels in loader:

            # Convert input and labels
            inputs = sample.reshape(-1, self.seq_len,
                                   self.input_dim).to(device)
            inputs = Variable(
                inputs, volatile=False, requires_grad=True).float()

            labels = labels.type(torch.int64).to(device)

            # Random baselines for IntegratedGradients, GradientShape and Occlusion
            baseline_single = torch.from_numpy(
                np.random.random(inputs.shape)).to(device)

            baseline_multiple = torch.from_numpy(np.random.random(
                (inputs.shape[0] * 5, self.seq_len, self.input_dim))).to(device)

            # Permutation feature mask
            mask = np.zeros((self.seq_len, self.input_dim), dtype=int)
            for i in range(self.seq_len):
                mask[i, :] = i

            mask_single = torch.from_numpy(mask).to(device)
            mask_single = mask_single.reshape(
                1, self.seq_len, self.input_dim).to(device)

            if method == "IG":
                IG = IntegratedGradients(self.model)
                e = IG.attribute(inputs, baselines=baseline_single, target=labels) \
                    .detach().cpu().numpy()

            elif method == "GS":
                GS = GradientShap(self.model)
                e = GS.attribute(inputs, baselines=baseline_multiple,
                                 stdevs=0.09,
                                 target=labels).detach().cpu().numpy()
            elif method == "FP":
                FP = FeaturePermutation(self.model)
                e = FP.attribute(inputs, target=labels,
                                 perturbations_per_eval=inputs.shape[0]).detach().cpu().numpy()
            elif method == "FA":
                FA = FeatureAblation(self.model)
                e = FA.attribute(inputs, target=labels).detach().cpu().numpy()

            elif method == "OS":
                OS = Occlusion(self.model)
                e = OS.attribute(inputs,
                                 sliding_window_shapes=(2, self.input_dim),
                                 target=labels,
                                 baselines=baseline_single).detach().cpu().numpy()
            elif method == "DM":
                model = self.model

                e = []
                areas = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
                for input in tqdm(inputs):
                    pert = GaussianBlur(device)
                    mask = MaskGroup(pert, device, verbose=False)
                    mask.fit(input, model, loss_function=mse, area_list=areas, size_reg_factor_init=0.01, n_epoch=100)
                    sample_e = mask.masks_tensor.detach().cpu().numpy()
                    e.append(sample_e)
            else:
                raise Exception("Provided explanation method is not implemented in the model wrapper class.")

            explanations.extend(e)

        explanations = np.array(explanations)
        explanations = np.swapaxes(explanations,-1,-2)
        return explanations

    def summary(self):
        print(self.model)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)
