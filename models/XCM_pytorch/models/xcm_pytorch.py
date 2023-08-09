import torch.nn as nn
import torch.nn.functional as F
from torch import reshape, cat
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils import data
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


class TSDataset(data.Dataset):
    def __init__(self, x_input, labels):
        self.samples = x_input
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class XCM_pyTorch(nn.Module):
    def __init__(self, n, k, window_size, n_class, filters_num=128):

        self.n = n
        self.k = k
        self.n_class = n_class

        super(XCM_pyTorch, self).__init__()
        self.conv1 = nn.Conv2d(1, int(filters_num), kernel_size=(int(window_size * self.n),1), stride=(1, 1), padding="same")
        self.conv1_batchNorm = nn.BatchNorm2d(int(filters_num))
        self.conv2 = nn.Conv2d(int(filters_num), 1, kernel_size=(1, 1), stride=(1, 1), padding="valid")

        self.convTemp_1 = nn.Conv1d(self.k, int(filters_num), kernel_size=int((window_size * self.n)), stride=1,
                                    padding="same")
        self.convTemp_1_batchNorm = nn.BatchNorm1d(int(filters_num))
        self.convTemp_2 = nn.Conv1d(int(filters_num), 1, kernel_size=1, stride=1)

        self.convFinal = nn.Conv1d(self.k + 1, int(filters_num), kernel_size=int(window_size * self.n), stride=1,
                                   padding="same")
        self.convFinal_batchNorm = nn.BatchNorm1d(int(filters_num))

        self.GAP = nn.AvgPool1d(self.n)
        self.fc1 = nn.Linear(int(filters_num),  self.n_class)

        self.recordBackwardGradients = None
        self.gradients = None

    def storeGradientsConvolution(self, type):
        self.recordBackwardGradients = type
        if self.recordBackwardGradients == None:
            self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        # parallel pipelines:

        ## 2D CONV
        single_input = False
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, 0) # Single input handling
            single_input = True
        x_1 = reshape(x, (x.shape[0], 1, self.n, self.k))
        x_ResFirstConv = F.relu(self.conv1_batchNorm(self.conv1(x_1)))
        x_1 = F.relu((self.conv2(x_ResFirstConv)))

        if self.recordBackwardGradients == '2D':
            x_ResFirstConv.register_hook(self.save_gradient)


         ## 1D CONV
        x_2 = reshape(x, (x.shape[0], self.k, self.n))
        x_ResFirstConvTemporal = F.relu(self.convTemp_1_batchNorm(self.convTemp_1(x_2)))
        x_2 = F.relu((self.convTemp_2(x_ResFirstConvTemporal)))
        x_2 = reshape(x_2, (x_2.shape[0], x_2.shape[1], x_2.shape[2], 1))

        if self.recordBackwardGradients == '1D':
            x_ResFirstConvTemporal.register_hook(self.save_gradient)

        ## FINAL STEP 1D CONV + GAP + FCL + SOFTMAX

        z = cat((x_1, x_2), 3)
        z = reshape(z, (z.shape[0], z.shape[3], z.shape[2]))
        z = F.relu(self.convFinal_batchNorm(self.convFinal(z)))

        z = self.GAP(z)
        z = reshape(z, (z.shape[0], z.shape[1]))
        z = self.fc1(z)

        return F.softmax(z,dim=1) if not single_input else torch.squeeze(F.softmax(z,dim=1))



class ModelCNN():
    def __init__(self,
                 model,
                 device="cpu",
                 criterion=nn.CrossEntropyLoss(),
                 n_epochs_stop=500,
                 learning_rate=0.001,
                 verbose=False):
        self.model = model.to(device)
        self.n_epochs_stop = n_epochs_stop
        self.criterion = criterion
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.device = device
        self.verbose = verbose

    def test(self, dataloader):
        mean_loss = []
        mean_accuracy = []
        total_sample = []

        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                self.model.eval()
                ts, label = batch_data
                img = Variable(ts.float()).to(self.device)
                v_label = Variable(label.float()).to(self.device)
                # ===================forward=====================
                output = self.model(img.float()).to(self.device)
                loss = self.criterion(output.float(), v_label.long())

                # ================eval on test===================
                total = label.size(0)
                _, predicted = torch.max(output.data, 1)
                #, label_train = torch.max(label.data, 1)
                #correct = (predicted.to(self.device) == label_train.to(self.device)).sum().item()
                correct = (predicted.to(self.device) == label.to(self.device)).sum().item()

                mean_loss.append(loss.item())
                mean_accuracy.append(correct)
                total_sample.append(total)

        return mean_loss, mean_accuracy, total_sample



    def train(self, num_epochs, dataloader_cl1, dataloader_cl1_test = [],  dataloader_cl1_validation= [] , model_name='model'):
        epochs_no_improve = 0
        min_val_loss = np.Inf

        loss_val_history = []
        loss_train_history = []
        loss_test_history = []
        accuracy_test_history = []
        accuracy_val_history = []

        for epoch in range(num_epochs):
            mean_loss_train = []
            mean_accuracy_train = []
            total_sample_train = []

            for i, batch_data_train in enumerate(dataloader_cl1):
                self.model.train()

                ts_train, label_train = batch_data_train

                img_train = Variable(ts_train.float()).to(self.device)
                v_label_train = Variable(label_train.float()).to(self.device)

                # ===================forward=====================
                self.optimizer.zero_grad()
                output_train = self.model(img_train.float()).to(self.device)

                # ===================backward====================
                loss_train = self.criterion(output_train.float(), v_label_train.long())
                loss_train.backward()
                self.optimizer.step()

                # ================eval on train==================
                total_train = label_train.size(0)
                _, predicted_train = torch.max(output_train.data, 1)

                correct_train = (predicted_train.to(self.device) == label_train.to(self.device)).sum().item()
               # _, label_train = torch.max(label_train.data, 1)
               # correct_train = (predicted_train.to(self.device) == label_train.to(self.device)).sum().item()

                mean_loss_train.append(loss_train.item())
                mean_accuracy_train.append(correct_train)
                total_sample_train.append(total_train)

            # ==================eval on test=====================
            mean_loss_test, mean_accuracy_test, total_sample_test = self.test(dataloader_cl1_test)

            # ==================eval on validation=====================
            mean_loss_val, mean_accuracy_val, total_sample_val = self.test(dataloader_cl1_validation)

            # ====================verbose========================
            if self.verbose:
                if epoch % 10 == 0:
                    print(
                        'Epoch [{}/{}], Loss Train: {:.4f},Loss Test: {:.4f}, Loss Val: {:.4f},Accuracy Train: {:.2f}%, Accuracy Test: {:.2f}%, Accuracy Val: {:.2f}%'
                        .format(epoch + 1,
                                num_epochs,
                                np.mean(mean_loss_train),
                                np.mean(mean_loss_test),
                                np.mean(mean_loss_val),
                                (np.sum(mean_accuracy_train) / np.sum(total_sample_train)) * 100,
                                (np.sum(mean_accuracy_test) / np.sum(total_sample_test)) * 100,
                                (np.sum(mean_accuracy_val) / np.sum(total_sample_val)) * 100))

                # ======================log==========================
                loss_test_history.append(np.mean(mean_loss_test))
                loss_train_history.append(np.mean(mean_loss_train))
                loss_val_history.append(np.mean(mean_loss_val))

                accuracy_test_history.append(np.sum(mean_accuracy_test) / np.sum(total_sample_test))
                accuracy_val_history.append(np.sum(mean_accuracy_val) / np.sum(total_sample_val))

                self.loss_test_history = loss_test_history
                self.loss_train_history = loss_train_history
                self.accuracy_test_history = accuracy_test_history
                self.accuracy_val_history = accuracy_val_history

                # ================early stopping=====================
                if epoch == 3:
                    min_val_loss = np.sum(mean_loss_test)

                if np.sum(mean_loss_test) < min_val_loss:
                    torch.save(self.model, model_name)
                    epochs_no_improve = 0
                    min_val_loss = np.sum(mean_loss_test)

                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == self.n_epochs_stop:
                        self.model = torch.load(model_name)
                        break

    def grad_cam_Pytorch(self, ts_instance, conv_type= "2D", produce_plot = False, label = None):
        """
        Grad-CAM output

        Parameters
        ----------
        ts_instance: numpy array
            MTS sample

        conv_type: string
            Type of the convolution layer

        Returns
        -------
        heatmap: array
            Heatmap
        """

        #self.model.eval()





        # prepare the instance tensor to explain ##############
        instanceT= torch.tensor(ts_instance, requires_grad=True)
        instance = instanceT.reshape(1,instanceT.shape[0],instanceT.shape[1],instanceT.shape[2]).float().to(self.device)
        #######################################################



        # create the hook, which gets the output of the first 1D or 2D convolutional layer########
        convLayerOut = None

        if conv_type == "1D":
            def First1DConvLayerOut_hook(module, input_, output):
                nonlocal convLayerOut
                convLayerOut = output

            self.model.convTemp_1_batchNorm.register_forward_hook(First1DConvLayerOut_hook)
        else:
            def First2DConvLayerOut_hook(module, input_, output):
                nonlocal convLayerOut
                convLayerOut = output

            self.model.conv1_batchNorm.register_forward_hook(First2DConvLayerOut_hook)

        ###################################################################################

        self.model.storeGradientsConvolution(conv_type)
        predictions = self.model(instance)
        pred_index = torch.argmax(predictions)

        # torch.autograd version
        #pred_class = predictions[:, pred_index]
         # gradients = torch.autograd.grad(pred_class, convLayerOut, grad_outputs=torch.ones_like(pred_class),retain_graph=True)[0]

        one_hot_output = torch.FloatTensor(1, predictions.size()[-1]).zero_()
        one_hot_output[0][pred_index] = 1

        predictions.backward(gradient=one_hot_output.to(self.device), retain_graph=True)
        gradients = self.model.gradients
        self.model.storeGradientsConvolution(None)

        if not (label is None):
            correctClass = np.argmax(label)
            correctlyClassified = (correctClass == pred_index )
            if correctlyClassified and self.verbose:
                print("Explained example correctly classified")
       

        heatmap = None

        # Compute a weighted combination between the feature maps
        if conv_type == "1D":

            pooled_grads = torch.mean(gradients, dim=(0,2))
            conv_layer_output = convLayerOut[0].reshape(convLayerOut[0].shape[1],
                                                        convLayerOut[0].shape[0])
            heatmap = conv_layer_output * pooled_grads
            heatmap = torch.mean(heatmap, dim=1)
            heatmap = heatmap.reshape(1, heatmap.shape[0], 1, 1)

            upsample = nn.Upsample(size=(1, instance.shape[2]))
            heatmap = upsample(heatmap)


            heatmap = np.squeeze(heatmap.detach().numpy())


        else:
            pooled_grads = torch.mean(gradients, dim=(0, 2, 3))
            conv_layer_output = convLayerOut[0].reshape(convLayerOut[0].shape[1],
                                                               convLayerOut[0].shape[2],
                                                               convLayerOut[0].shape[0])

            heatmap = conv_layer_output * pooled_grads
            heatmap = torch.mean(heatmap, dim=2)
            heatmap = heatmap.detach().cpu().numpy()

        # Keep positive values
        heatmap = np.maximum(heatmap, 0)

        if (produce_plot):
            plt.figure(figsize=(25, 10))
            heatmap = np.swapaxes(normalize(heatmap), 0, 1)
            xticklabels = range(1, instance.shape[1] + 1)
            yticklabels = range(1, instance.shape[2] + 1)

            sns.heatmap(
                heatmap, xticklabels=xticklabels, yticklabels=yticklabels, cmap="RdBu_r"
            )
            plt.show()


        return heatmap


    def predict(self,data_test, n_class):
        self.model.eval()
        img = Variable(torch.from_numpy(data_test).float().to(self.device))

        output = self.model(img.float())
        output = output.detach().cpu().numpy()
        y_pred = np.argmax(output, axis=-1)
        if n_class == 2:
            y_score = output[:, 1]
            return y_score, y_pred
        else:
            return output, y_pred
