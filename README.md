# ExplCausalLens

## Abstract

Explainable machine learning techniques (XAI) aim to provide a solid descriptive approach to Deep Neural Networks (NN). In Multi-Variate Time Series (MTS) analysis, the most recurrent techniques use relevance attribution, where importance scores are assigned to each TS variable over time according to their importance in classification or forecasting. Despite their popularity, post-hoc explanation methods do not account for causal relationships between the model outcome and its predictors. In our work, we conduct a thorough empirical evaluation of model-agnostic and model-specific relevance attribution methods proposed for TCNN, LSTM, and Transformers classification models of MTS. The contribution of our empirical study is three-fold: (i) evaluate the capability of existing post-hoc methods to provide consistent explanations for high-dimensional MTS (ii) quantify how post-hoc explanations are related to sufficient explanations (i.e., the direct causes of the target TS variable) underlying  the datasets, and (iii) rank the performance of surrogate models built over post-hoc and causal explanations w.r.t. the full MTS models. 

 
To the best of our knowledge, this is the first work that evaluates the reliability and effectiveness of existing xAI methods from a temporal causal model perspective.

## Structure

- `data` contains the 3 dataset families used.
  - `SynthNonlin/7ts2h`: the 7ts2h dataset
  - `TestCLIM_N-5_T-250`: the CLIM dataset
  - `fMRI_processed_by_Nauta`: the fMRI dataset
- `dynamask`: contains the DynaMask method source files
- `experiments` contains the scripts to run experiments and plot results
  - `configs`: the configuration files given to the experiments
  - `results`: scripts to plot the results and .csv files containing our results
- `hyperparameters` contains the scripts for hyperparameters tuning
- `models` contains the script defining each of the 4 DNN architecture

## Requirements

We used the following configuration:
- CUDA Version 11.8, Driver Version 520.61.05
- torch==1.13.1
- scikit-learn==1.2.1
- scipy==1.10.0
- pandas==1.5.3
- numpy==1.24.1
- networkx==3.0
- seaborn==0.12.2
- imbalanced-learn==0.10.1
- optuna==3.1.0
- autorank==1.1.3

See requirements.txt for the whole trace of the python environment we used.

## Usage

To run the experiments with a specific model architecture on a specific dataset, the following commands can be launched successively:
- `cd experiments`
- `python3 experiment.py <config_name_without_extension>`, with argument for instance `lstm_clim_2`.
- `python3 evaluate_metrics.py <config_name_without_extension>` with same config name as previous command

To obtain the results in graphical form, launch a jupyter notebook in the subdirectory results:
- `cd results`
- `jupyter notebook`
Then successively run all cells of
1) `merge_results.ipynb`
2) `graph_plotting.ipynb`
3) `multiple hypothesis testing.ipynb`

