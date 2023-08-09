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
sys.path.insert(0, '../models/dCAM/src/models')
sys.path.insert(0, '../models/dCAM/src/explanation')

from CNN_models import TSDataset,ModelCNN
from DCAM import *

from torchsummary import summary
import torch
from torch import nn



class ConvNet2D(nn.Module):
    def __init__(self,original_length,original_dim,nb_channel,num_classes=10,TRIAL_number_of_layers=6,TRIAL_number_of_channels=128,TRIAL_time_kernel=3):
        super(ConvNet2D, self).__init__()
        
        self.TRIAL_number_of_layers=TRIAL_number_of_layers
        self.TRIAL_number_of_channels=TRIAL_number_of_channels
        self.TRIAL_time_kernel = TRIAL_time_kernel
        self.kernel_size = (1,self.TRIAL_time_kernel)
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
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels, kernel_size = self.kernel_size, padding=self.padding),      
                nn.ReLU(),
                )
        else:
            self.layer2 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels, kernel_size = self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels),        
                nn.ReLU(),
                )
        
        if self.TRIAL_number_of_layers == 3:
            self.layer21 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels*2, kernel_size = self.kernel_size, padding=self.padding),
                nn.ReLU(),
                )
        else:
            self.layer21 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels, TRIAL_number_of_channels*2, kernel_size = self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels*2),
                nn.ReLU(),
                )
        
        if self.TRIAL_number_of_layers == 4:
            self.layer22 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels*2, TRIAL_number_of_channels*2, kernel_size = self.kernel_size, padding=self.padding),
                nn.ReLU(),
                )
        else:
            self.layer22 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels*2, TRIAL_number_of_channels*2, kernel_size = self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels*2),
                nn.ReLU(),
                )
            
        if self.TRIAL_number_of_layers == 5:
            self.layer23 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels*2, TRIAL_number_of_channels*2, kernel_size = self.kernel_size, padding=self.padding),
                nn.ReLU(),
                )
        else:
            self.layer23 = nn.Sequential(
                nn.Conv2d(TRIAL_number_of_channels*2, TRIAL_number_of_channels*2, kernel_size = self.kernel_size, padding=self.padding),
                nn.BatchNorm2d(TRIAL_number_of_channels*2),
                nn.ReLU(),
                )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(TRIAL_number_of_channels*2,TRIAL_number_of_channels*2, kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            )
            
            
        self.GAP = nn.AvgPool2d(kernel_size=(original_dim,original_length))
        
        outshape = [TRIAL_number_of_channels,TRIAL_number_of_channels,TRIAL_number_of_channels*2,TRIAL_number_of_channels*2,TRIAL_number_of_channels*2,TRIAL_number_of_channels*2][self.TRIAL_number_of_layers-1]
        self.fc1 = nn.Sequential(nn.Linear(outshape,num_classes))
        
        
    
    def forward(self, x):
        if self.TRIAL_number_of_layers>0:
            out = self.layer1(x)
        if self.TRIAL_number_of_layers>1:
            out = self.layer2(out)
        if self.TRIAL_number_of_layers>2:
            out = self.layer21(out)
        if self.TRIAL_number_of_layers>3:
            out = self.layer22(out)
        if self.TRIAL_number_of_layers>4:
            out = self.layer23(out)
        if self.TRIAL_number_of_layers>5:
            out = self.layer3(out)
        
        out = self.GAP(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out









class model_constructor():
    def gen_cube(self,instance):
        #rotate the features index to obtain the right format
        instance = instance.T
        result = []
        for i in range(len(instance)):
            result.append([instance[(i+j)%len(instance)] for j in range(len(instance))])
        return np.array(result)
    
    def _prepare_data_train(self,data,labels,shuffle=True):
        all_class = np.array([self.gen_cube(acl) for acl in data])
        dataset_mat = TSDataset(all_class,labels)
        loader = torch.utils.data.DataLoader(dataset_mat, batch_size=self.batch_size, shuffle=shuffle)
        return loader
    
    def _prepare_data_test(self,data):
        all_class = np.array([self.gen_cube(acl) for acl in data])
        dataset_mat = TSDataset(all_class,np.zeros((all_class.shape[0],)))
        loader = torch.utils.data.DataLoader(dataset_mat, batch_size=self.batch_size, shuffle=False)
        return loader
    
    
    def __init__(self,device,original_length,original_dim,num_classes,batch_size,num_epochs,nb_permutations,TRIAL_number_of_layers=6,TRIAL_number_of_channels=128,TRIAL_time_kernel=3):
        self.device = device
        self.original_length = original_length
        self.original_dim = original_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.nb_permutations = nb_permutations
        
        modelarch = ConvNet2D(self.original_length,self.original_dim,self.original_dim,self.num_classes,
                              TRIAL_number_of_layers=TRIAL_number_of_layers,TRIAL_number_of_channels=TRIAL_number_of_channels,TRIAL_time_kernel=TRIAL_time_kernel).to(self.device)
        self.model = ModelCNN(modelarch,self.device,verbose=False)
        
    def train(self,data_train,labels_train):
        loader = self._prepare_data_train(data_train,labels_train,shuffle=True)
        self.model.train(self.num_epochs,dataloader_cl1=loader,dataloader_cl1_test = [])
        
    def predict(self,data_test):
        loader = self._prepare_data_test(data_test)
        
        y_score = []
        with torch.no_grad():
            for batch in loader:
                self.model.model.eval()
                batch,_ = batch
                batch = torch.autograd.Variable(batch.float()).to(self.device)
                output = self.model.model(batch.float()).to(self.device)
                non_normalized_out = output.data.detach().cpu().numpy()
                if self.num_classes==2:
                    normalized_proba = scipy.special.softmax(non_normalized_out,axis=1)[:,1]
                    y_score.append(normalized_proba)
        if self.num_classes==2:
            return np.concatenate(y_score)
    
    def explain(self,data_sample,true_label,pred_label):
        pass
    
    def summary(self):
        return summary(self.model.model.to(self.device),input_size=(self.original_dim,self.original_dim,self.original_length))
    



dataset_name = "SynthNonlin/7ts2h"
filename = "data_0.csv"


df = pd.read_csv("../data/"+dataset_name+"/"+filename)
df=df[df.columns[1:]]
VAR_NAMES = list(df.columns)

pastPointsToForecast = 8
numberFolds=3


#prepare data

final_data = defaultdict(list)

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
        
        final_data[TARGET_NAME].append([data_train,labels_train,data_test,labels_test])







def objective(trial):
    
    TRIAL_number_of_layers = trial.suggest_int("nlayers",3,6)
    TRIAL_number_of_channels = trial.suggest_int("nchannels",50,128,log=True)
    TRIAL_time_kernel = trial.suggest_int("timekernel",2,4)
    TRIAL_num_epochs = trial.suggest_int("num_epochs",10,40,15)
    
    optimize_score = 0

    for TARGET_NAME in VAR_NAMES:
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        original_length = pastPointsToForecast
        original_dim = len(VAR_NAMES)
        num_classes = 2
        batch_size = 32
        num_epochs = TRIAL_num_epochs
        nb_permutations = 200
        config = {"device":device,"original_length":original_length,"original_dim":original_dim,"num_classes":num_classes,"batch_size":batch_size,"num_epochs":TRIAL_num_epochs,"nb_permutations":nb_permutations,
        "TRIAL_number_of_layers":TRIAL_number_of_layers,"TRIAL_number_of_channels":TRIAL_number_of_channels,"TRIAL_time_kernel":TRIAL_time_kernel}
        
        
        y_true = []
        y_score = []

        for data_train,labels_train,data_test,labels_test in final_data[TARGET_NAME]:
            
            #insert model declaration here
            model = model_constructor(**config)
            #print(model.summary())
            
            #insert model training here
            model.train(data_train,labels_train) #replace as necessary
            
            #obtain predicted labels from the test set
            score_pred = model.predict(data_test) #replace as necessary
            
            
                
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
    s="\ndCAM model, "+dataset_name+"/"+filename
    s=s+"\nNumber of trials:"+str(n_trials)
    s=s+"\nTime spent:"+str(int(time.time()-begin_time))
    s=s+"\n"+str(study.best_params)+"\n"+"optimized_goal: "+str(study.best_value)
    s=s+"\n\n"
    f.write(s)


    
    
