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



rng = np.random.default_rng(0) # random seed
legacy_rng = np.random.RandomState(0)

#ADD ALL NECESSARY LIBRARIES FOR THE EXPERIMENT


rootdir = "../"

sys.path.insert(0, rootdir+'/models/dCAM/src/models')
import CNN_models


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

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.autograd import Variable
import torch.utils.data as data_utils



sys.path.insert(0, rootdir+'/models/LSTM_TSIB/')

import LSTM








class LSTM_model_constructor():
    
    def __init__(self,device, epochs, batch_size, input_size, hidden_size, num_classes, rnndropout, original_length):
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.rnndropout = rnndropout
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.original_dim = self.input_size
        self.original_length = original_length
        
        self.model = LSTM.LSTM(self.device,self.input_size,self.hidden_size,self.num_classes,self.rnndropout)
        
        
    def train(self,data_train,labels_train):
        data_train = CNN_models.TSDataset(data_train,labels_train)
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
            
        
    def predict(self,data_test):
        loader = torch.utils.data.DataLoader(data_test, batch_size=self.batch_size, shuffle=False)
        
        y_score = []
        y_pred = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                
                batch = torch.autograd.Variable(batch.float()).to(self.device)
                output = self.model(batch.float()).to(self.device)
                non_normalized_out = output.data.detach().cpu().numpy()
                y_pred.append(np.argmax(non_normalized_out,axis=-1))
                if self.num_classes==2:
                    normalized_proba = scipy.special.softmax(non_normalized_out,axis=1)[:,1]
                    y_score.append(normalized_proba)
                
        if self.num_classes==2:
            return np.concatenate(y_score),np.concatenate(y_pred)
    
    def explain(self,data_sample,true_label,pred_label,method):
        #we cannot use original model as the input is of shape (batch,D,D,T): explanations would be (D,D,T) too.
        self.explained_model = self.model
        explained_model = self.explained_model
        explained_model.eval()
        
        torch.backends.cudnn.enabled=False # seems like cudnn does not support gradient computation in eval mode for RNN... 
        
        loader = CNN_models.TSDataset(data_sample,pred_label)
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
                (inputs.shape[0]*5, inputs.shape[1], inputs.shape[2]))).float().to(device)

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
                                 perturbations_per_eval=inputs.shape[0],
                                 feature_mask=mask_single).detach().cpu().numpy()

            elif method == "FA":
                FA = FeatureAblation(explained_model)
                e = FA.attribute(
                    inputs, target=label).detach().cpu().numpy()

            elif method == "OS":
                OS = Occlusion(explained_model)
                e = OS.attribute(inputs,
                                 sliding_window_shapes=(1, self.original_dim),
                                 target=label,
                                 baselines=baseline_single).detach().cpu().numpy()

            explanations.extend(e)
        
        explanations = np.array(explanations)
        explanations = np.swapaxes(explanations,-1,-2) #the code standard is to get explanations of the form (D,T), but here we get (T,D).
        
        torch.backends.cudnn.enabled=True #setting back cudnn
        
        return np.array(explanations)
    
    def summary(self):
        return summary(self.model.to(self.device),input_size=(self.batch_size,None,self.input_size))
    
    def save(self,path):
        torch.save(self.model,path)
    def load(self,path):
        self.model = torch.load(path)

model_constructor = LSTM_model_constructor



dataset_name = "TestCLIM_N-5_T-250/returns"
#files where perf aren't bad
filenames = ["TestCLIM_N-5_T-250_0035.txt","TestCLIM_N-5_T-250_0002.txt","TestCLIM_N-5_T-250_0139.txt"]


final_data = defaultdict(list)

for filename in filenames:

    df = pd.read_csv("../data/"+dataset_name+"/"+filename,header=None,sep=" ")
    df.columns = [str(i) for i in df.columns]
    VAR_NAMES = list(df.columns)

    pastPointsToForecast = 8
    numberFolds=3


    #prepare data


    for TARGET_NAME in VAR_NAMES:
        data_for_labels = df[TARGET_NAME].values.reshape((-1,1))

        est = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans')
        est.fit(data_for_labels)
        labels = est.transform(data_for_labels)
        labels=labels.reshape((-1,))
        
        window_data = []
        window_labels = []

        for windowbegin in range(df.values.shape[0]-pastPointsToForecast):
            current_window = df.values[windowbegin:windowbegin+pastPointsToForecast,:]
            current_label = labels[windowbegin+pastPointsToForecast]
            window_data.append(current_window)
            window_labels.append(current_label)

        window_data = np.array(window_data)
        window_labels = np.array(window_labels)
        
        #EXCLUDE THE LATEST 30% FRAMES THAT MAKE THE TEST SET OF THE EXPERIMENTS
        separation = int((1-0.3)*len(window_data))
        window_data = window_data[:separation]
        window_labels = window_labels[:separation]
        
        #create folds
        skf_object = sklearn.model_selection.StratifiedKFold(numberFolds,shuffle=True,random_state=legacy_rng)
        skf_generator = skf_object.split(window_data, window_labels)

        
        for ith_fold,(train_index,test_index) in enumerate(skf_generator):
            data_train = window_data[train_index]
            labels_train = window_labels[train_index]
            data_test = window_data[test_index]
            labels_test = window_labels[test_index]
            
            #insert oversampling here
            new_index,labels_train = RandomOverSampler(random_state=legacy_rng).fit_resample(np.arange(len(data_train)).reshape((-1,1)),labels_train)
            data_train = data_train[new_index.reshape((-1,))]
            
            final_data[TARGET_NAME+filename].append([data_train,labels_train,data_test,labels_test])







def objective(trial):
    
    TRIAL_hidden = trial.suggest_int("hidden_size",16,256,log=True)
    TRIAL_dropout = trial.suggest_float("rnndropout",0.2,0.5)
    TRIAL_epochs = trial.suggest_int("epochs",10,40,10)
    
    optimize_score = 0

    for TARGET_NAME in final_data:
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        original_length = pastPointsToForecast
        input_size = final_data[TARGET_NAME][0][0].shape[-1]
        num_classes = 2
        batch_size = 32
        epochs = TRIAL_epochs
        rnndropout = TRIAL_dropout
        hidden_size = TRIAL_hidden
        config = {"device":device,"epochs":epochs, "batch_size":batch_size, "input_size":input_size, "hidden_size":hidden_size, "num_classes":num_classes, "rnndropout":rnndropout, "original_length":original_length}
        
        
        y_true = []
        y_score = []

        for data_train,labels_train,data_test,labels_test in final_data[TARGET_NAME]:
            
            #insert model declaration here
            model = model_constructor(**config)
            #print(model.summary())
            
            #insert model training here
            model.train(data_train,labels_train) #replace as necessary
            
            #obtain predicted labels from the test set
            score_pred, labels_pred = model.predict(data_test) #replace as necessary
            
            
                
            #keep track of results
            y_score.append(score_pred)
            y_true.append(labels_test)

        #convert to array
        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        
        #balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true=y_true,y_pred=y_pred)
        roc_auc_score = sklearn.metrics.roc_auc_score(y_true,y_score)

        optimize_score += roc_auc_score
        
        
    return -optimize_score
        
n_trials=100
study = optuna.create_study()
begin_time=time.time()
study.optimize(objective,n_trials=n_trials)
print("Time spent",time.time()-begin_time)
print(study.best_params)
print(study.best_value)
    
with open("results_rocauc.txt","a") as f:
    s="\nLSTM model, "+dataset_name+"/"+filename
    s=s+"\nNumber of trials:"+str(n_trials)
    s=s+"\nTime spent:"+str(int(time.time()-begin_time))
    s=s+"\n"+str(study.best_params)+"\n"+"optimized_goal: "+str(study.best_value)
    s=s+"\n\n"
    f.write(s)


    
    
