import sys
import time
import os

import numpy as np
import scipy.special
import pandas as pd

import optuna

from torchsummary import summary
import torch
from torch import nn


import scipy.stats
import scipy.special
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from imblearn.over_sampling import RandomOverSampler

from collections import defaultdict


from torch.autograd import Variable
import torch.utils.data as data_utils

rootdir = "../"
sys.path.insert(0, rootdir + '/models')

from XCM_pytorch.models.xcm_pytorch import XCM_pyTorch, ModelCNN, TSDataset



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



rng = np.random.default_rng(0) # random seed
legacy_rng = np.random.RandomState(0)




class XCM_model_constructor:

    def _prepare_training_data(self, data, labels, batchsize):
        dataset = TSDataset(np.array(data), np.array(labels))
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

    def summary(self):
        return summary(self.model.model.to(self.device),input_size=self.input_shape)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)

model_constructor = XCM_model_constructor





dataset_name = "fMRI_processed_by_Nauta/returns/our_selection"

filename = "timeseries1.csv"


df = pd.read_csv("../data/"+dataset_name+"/"+filename)
#df=df[df.columns[1:]]
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
    
    df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    
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
        
        final_data[TARGET_NAME].append([data_train,labels_train,data_test,labels_test])







def objective(trial):
    
    TRIAL_filters_num = trial.suggest_int("filters_num",32,256,log=True)
    TRIAL_window_size = trial.suggest_float("time_window",0.2,0.6)
    TRIAL_epochs = trial.suggest_int("epochs",5,40,5)
    
    optimize_score = 0

    for TARGET_NAME in VAR_NAMES:
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        input_shape = (pastPointsToForecast,len(VAR_NAMES),1)
        num_classes = 2
        batch_size = 32
        epochs = TRIAL_epochs
        window_size = TRIAL_window_size
        filters_num = TRIAL_filters_num
        config = {"device":device, "input_shape":input_shape, "n_class":num_classes, "window_size":window_size, "filters_num":filters_num, "batch_size":batch_size, "epochs":epochs}
        
        
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
    s="\nXCM_pytorch model, "+dataset_name+"/"+filename
    s=s+"\nNumber of trials:"+str(n_trials)
    s=s+"\nTime spent:"+str(int(time.time()-begin_time))
    s=s+"\n"+str(study.best_params)+"\n"+"optimized_goal: "+str(study.best_value)
    s=s+"\n\n"
    f.write(s)


    
    














    
    
