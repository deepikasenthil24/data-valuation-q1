import copy
import tqdm
import optuna
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_dataset import load
from multiprocessing import Pool, freeze_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mutual_info_score, auc, roc_curve, roc_auc_score, f1_score, accuracy_score
from scipy.stats import wasserstein_distance
from optuna.samplers import *
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from API_Design_a import MissingValueError, LabelError, SamplingError, Injector
from boostclean import boost_clean
import torch
import torch.nn as nn

import argparse
import os
import json

params = {
            "num_epochs": 100,
            "batch_size": 512,
            "device": "cpu",
            "model_lr": 0.01,
            "weight_decay": 0,
            "model": "log",
            "train_seed": 1,
            "split_seed": 1,
            "method": "diffprep_fix",
            "save_model": True,
            "logging": False,
            "no_crash": False,
            "patience": 3,
            "momentum": 0.9,
            "prep_lr": None,
            "temperature": 0.1,
            "grad_clip": None,
            "pipeline_update_sample_size": 512,
            "init_method": "default",
            "diff_method": "num_diff",
            "sample": False
        }

def initialize_model_and_baseline_auc(dataset, model_name, cleaning, sens_attr):
    X_train, X_test, y_train, y_test = load(dataset)
    # with open('data_overview.txt', 'w') as f:
    #     f.write("X_train Information:\n")
    #     f.write(f"Shape: {X_train.shape}\n")
    #     f.write(f"Head:\n{X_train.head()}\n\n")
    #
    #     f.write("X_test Information:\n")
    #     f.write(f"Shape: {X_test.shape}\n")
    #     f.write(f"Head:\n{X_test.head()}\n\n")
    #
    #     f.write("y_train Information:\n")
    #     f.write(f"Shape: {y_train.shape}\n")
    #     f.write(f"Head:\n{y_train.head()}\n\n")
    #
    #     f.write("y_test Information:\n")
    #     f.write(f"Shape: {y_test.shape}\n")
    #     f.write(f"Head:\n{y_test.head()}\n\n")
    X_train_orig = copy.deepcopy(X_train)
    X_test_orig = copy.deepcopy(X_test)
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(X_train_orig, y_train, test_size=0.25, random_state=42)

    X_train_orig, X_val_orig, X_test_orig = (X_train_orig.reset_index(drop=True),
                                             X_val_orig.reset_index(drop=True),
                                             X_test_orig.reset_index(drop=True))
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    if cleaning == 'boostclean':
        model = get_model(model_name)
        baseline_acc, baseline_auc, baseline_spd, baseline_eo, baseline_f1 = boost_clean(
            model, X_train_orig,
            y_train.to_numpy(), X_test_orig,
            y_test.to_numpy(),
            X_test_sensitive=X_test_orig[sens_attr].copy(), T=3)
    elif cleaning == 'h2o':
        ss = StandardScaler()
        ss.fit(X_train_orig)
        X_train = ss.transform(X_train_orig)
        X_test = ss.transform(X_test_orig)

        clf = get_model(model_name)
        clf.fit(X_train, y_train)

        # model_weights = clf.coef_
        # feature_names = X_train_orig.columns
        # plt.figure(figsize=(10, 8))
        # plt.barh(feature_names, model_weights[0])
        # plt.xlabel('Weight')
        # plt.ylabel('Feature')
        # plt.title('Logistic Regression Model Weights')
        # plt.savefig('model_weights.pdf', format='pdf')
        # plt.close()

        y_test_pred = clf.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, y_test_pred)
    elif cleaning in ['MeanImputer', 'MedianImputer', 'KNNImputer', 'IterativeImputer']:
        imputers = {
            'MeanImputer': SimpleImputer(strategy='mean'),
            'MedianImputer': SimpleImputer(strategy='median'),
            'KNNImputer': KNNImputer(n_neighbors=10),
            'IterativeImputer': IterativeImputer(max_iter=10, random_state=42)
        }
        imputer = imputers[cleaning]
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_orig), columns=X_train_orig.columns)
        model = get_model(model_name)
        model.fit(X_train_imputed, y_train)
        y_test_pred_proba = model.predict_proba(X_test_orig)[:, 1]
        baseline_auc = roc_auc_score(y_test, y_test_pred_proba)
    elif cleaning == 'autosklearn':
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn_add_custom_clfs import add_clf

        X_train, X_test, y_train, y_test = load(dataset)
        X_test_orig = copy.deepcopy(X_test)
        if model_name == 'DT':
            clf_name = 'CustomDecisionTree'
        elif model_name == 'RF':
            clf_name = 'CustomRandomForest'
        elif model_name == 'SVM':
            clf_name = 'CustomSVM'
        elif model_name == 'NN':
            clf_name = 'CustomMLPClassifier'
        elif model_name == 'LR':
            clf_name = 'CustomLogisticRegression'
        else:
            raise ValueError("Unsupported autosklearn model name")

        add_clf(clf_name)
        overall_time = 30
        per_run_time = overall_time / 5
        clf = AutoSklearnClassifier(time_left_for_this_task=overall_time, per_run_time_limit=per_run_time,
                                    include={'classifier': [clf_name]}, memory_limit=1024 * 8)
        clf.fit(X_train, y_train)
        y_test_pred_proba = clf.predict_proba(X_test_orig)[:, 1]
        baseline_auc = roc_auc_score(y_test, y_test_pred_proba)
    elif cleaning == 'random':
        from random_search import RandomSearch
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn_add_custom_clfs import add_clf

        if model_name == 'DT':
            clf_name = 'DecisionTree'
        elif model_name == 'RF':
            clf_name = 'RandomForest'
        elif model_name == 'SVM':
            clf_name = 'SVM'
        elif model_name == 'NN':
            clf_name = 'MLPClassifier'
        else:
            clf_name = 'LogisticRegression'

        fitted_clf, val_auc = RandomSearch(X_train_orig, y_train, X_val_orig, y_val,
                                           clf_name=clf_name, n_trials=10)
        baseline_auc = roc_auc_score(y_test, fitted_clf.predict_proba(X_test_orig)[:, 1])
    elif cleaning == 'diffprep':
        baseline_auc = roc_auc_score(y_test, DecisionTreeClassifier(random_state=42).fit(X_train_orig, y_train).predict_proba(X_test_orig)[:, 1])
    else:
        raise ValueError("Unsupported cleaning approach name")

    # correlation_matrix = pd.concat([X_train_orig, y_train], axis=1).corr()
    # plt.figure(figsize=(15, 15))
    # plt.matshow(correlation_matrix, fignum=1)
    # plt.xticks(range(correlation_matrix.shape[1]), correlation_matrix.columns, rotation=90)
    # plt.yticks(range(correlation_matrix.shape[1]), correlation_matrix.columns)
    # plt.colorbar()
    # plt.title('Feature Correlation Matrix', pad=20)
    # plt.savefig('feature_correlation_matrix.pdf', format='pdf')
    # plt.close()

    return baseline_auc

def get_model(model_name):
    if model_name == 'LR':
        return LogisticRegression(random_state=42)
    elif model_name == 'DT':
        return DecisionTreeClassifier(random_state=42)
    elif model_name == 'NN':
        return MLPClassifier(random_state=42, hidden_layer_sizes=(10,))
    elif model_name == 'RF':
        return RandomForestClassifier(random_state=42)
    elif model_name == 'SVM':
        sgd_clf = SGDClassifier(loss='hinge', random_state=42, max_iter=100, tol=1e-3)
        return CalibratedClassifierCV(sgd_clf, method='sigmoid', cv=5)
    else:
        raise ValueError("Unsupported model name")

def create_pattern(col_list, lb_list, ub_list):
    try:
        assert len(col_list) == len(lb_list) == len(ub_list)
    except:
        print(col_list, lb_list, ub_list)
        raise SyntaxError

    def pattern(data_X, data_y):
        binary_indicators = []
        for i in data_X.index:
            satisfaction = True
            for j in range(len(col_list)):
                if col_list[j] == 'Y':
                    if (data_y.loc[i] < lb_list[j]) or (data_y.loc[i] > ub_list[j]):
                        satisfaction = False
                        break
                else:
                    if (data_X.loc[i, col_list[j]] < lb_list[j]) or (data_X.loc[i, col_list[j]] > ub_list[j]):
                        satisfaction = False
                        break
            if satisfaction:
                binary_indicators.append(1)
            else:
                binary_indicators.append(0)
        return np.array(binary_indicators)

    return pattern

def objective_fuc(trial, col_id, budget, baseline_auc, pattern_col_list, error_type, dataset, model_name, objective, sens_attr, cleaning):
    #     t0 = time.time()

    lb_list = []
    ub_list = []
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()

    X_train_orig.reset_index(drop=True, inplace=True)
    X_test_orig.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    #     print(time.time()-t0)
    #     t0 = time.time()

    if error_type == 'MAR':
        pattern_col_list_copy = copy.deepcopy(pattern_col_list)
        excluded_col = X_train_orig.columns[col_id]
        if excluded_col in pattern_col_list_copy:
            pattern_col_list_copy.remove(excluded_col)
        current_pattern_col_list = pattern_col_list_copy
    else:  # Default to MNAR if not specified
        current_pattern_col_list = pattern_col_list

    def map_percent_to_value(data, percent):
        sorted_data = data.sort_values().reset_index(drop=True)
        percent = min(max(percent, 0), 1)
        index = int(percent * (len(sorted_data) - 1))
        return sorted_data.iloc[index]

    for col in current_pattern_col_list:
        if col == 'Y':
            if pd.api.types.is_integer_dtype(y_train):
                mv_lb = trial.suggest_int(col + '_mv_lb', int(y_train.min()), int(y_train.max()))
                mv_interval = trial.suggest_int(col + '_mv_int', 0, int(y_train.max()) - mv_lb)
            else:
                mv_lb = trial.suggest_float(col + '_mv_lb', y_train.min(), y_train.max())
                mv_interval = trial.suggest_float(col + '_mv_int', 0, y_train.max() - mv_lb)
            lb_list.append(mv_lb)
            mv_ub = mv_interval + mv_lb
            ub_list.append(mv_ub)
        else:
            if pd.api.types.is_integer_dtype(X_train_orig[col]):
                mv_lb = trial.suggest_int(col + '_mv_lb', int(X_train_orig[col].min()), int(X_train_orig[col].max()))
                mv_interval = trial.suggest_int(col + '_mv_int', 0, int(X_train_orig[col].max()) - mv_lb)
                mv_ub = mv_interval + mv_lb
            else:
                print("test")
                mv_lb_percent = trial.suggest_float(col + '_mv_lb', 0, 1)
                mv_interval_percent = trial.suggest_float(col + '_mv_int', 0, 1 - mv_lb_percent)
                mv_ub_percent = mv_interval_percent + mv_lb_percent
                # map precent to real value
                print(
                    f"Before mapping to actual values: mv_lb_percent: {mv_lb_percent}, mv_ub_percent: {mv_ub_percent}")
                mv_lb = map_percent_to_value(X_train_orig[col], mv_lb_percent)
                mv_ub = map_percent_to_value(X_train_orig[col], mv_ub_percent)
                print(f"After mapping to actual values: mv_lb: {mv_lb}, mv_ub: {mv_ub}")
            lb_list.append(mv_lb)
            ub_list.append(mv_ub)

    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            raise optuna.exceptions.TrialPruned()

    mv_pattern = create_pattern(current_pattern_col_list, lb_list, ub_list)
    mv_pattern_len = np.sum(mv_pattern(X_train_orig, y_train))

    if mv_pattern_len == 0:
        raise optuna.exceptions.TrialPruned()
    mv_num = min(mv_pattern_len, budget)

    if error_type == 'Label':
        label_flip_ratio = min(mv_pattern_len, budget) / mv_pattern_len
        mv_err = LabelError(pattern=mv_pattern, ratio=label_flip_ratio)
    elif error_type == 'Sampling':
        sampling_error_ratio = min(mv_pattern_len, budget) / mv_pattern_len
        mv_err = SamplingError(pattern=mv_pattern, ratio=sampling_error_ratio)
    else:
        mv_err = MissingValueError(col_id, mv_pattern, mv_num / mv_pattern_len)

    injecter = Injector(error_seq=[mv_err])
    dirty_X_train_orig, dirty_y_train, _, _ = injecter.inject(X_train_orig.copy(), y_train.copy(), X_train_orig,
                                                              y_train, seed=42)

    if len(np.unique(dirty_y_train)) < 2:
        raise optuna.exceptions.TrialPruned()
    trial.set_user_attr('num_errs', min(mv_pattern_len, budget))

    #     print(time.time()-t0)
    #     t0 = time.time()

    if cleaning == 'boostclean':

        model = get_model(model_name)
        acc, auc, spd, eo, f1 = boost_clean(model, dirty_X_train_orig,
                                            dirty_y_train.to_numpy(),
                                            X_test_orig, y_test.to_numpy(),
                                            X_test_sensitive=X_test_orig[sens_attr].copy(), T=3)
    elif cleaning == 'h2o':
        try:
            dirty_X_train_orig = SimpleImputer(strategy='mean').fit_transform(dirty_X_train_orig)
            ss = StandardScaler()
            ss.fit(dirty_X_train_orig)
            X_train = ss.transform(dirty_X_train_orig)
            X_test = ss.transform(X_test_orig)

            unique_train_labels = np.unique(dirty_y_train)
            unique_test_labels = np.unique(y_test)

            if len(unique_train_labels) == 1 or len(unique_test_labels) == 1:
                raise optuna.exceptions.TrialPruned()

            clf = get_model(model_name)
            clf.fit(X_train, dirty_y_train)

            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            if y_pred_proba.shape[1] == 1 and error_type == 'Sampling':
                raise optuna.exceptions.TrialPruned()
            else:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            spd = np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
                  np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
            eo = np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
                 np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
        except ValueError as e:
            if 'Requesting 5-fold cross-validation but provided less than 5 examples for at least one class' in str(e):
                raise optuna.exceptions.TrialPruned()
            else:
                raise e
    elif cleaning in ['MeanImputer', 'MedianImputer', 'KNNImputer', 'IterativeImputer']:
        imputers = {
            'MeanImputer': SimpleImputer(strategy='mean'),
            'MedianImputer': SimpleImputer(strategy='median'),
            'KNNImputer': KNNImputer(n_neighbors=10),
            'IterativeImputer': IterativeImputer(max_iter=10, random_state=42)
        }
        imputer = imputers[cleaning]
        dirty_X_train_imputed = pd.DataFrame(imputer.fit_transform(dirty_X_train_orig),
                                             columns=dirty_X_train_orig.columns)

        clf = get_model(model_name)
        clf.fit(dirty_X_train_imputed, dirty_y_train)

        y_pred = clf.predict(X_test_orig)
        y_pred_proba = clf.predict_proba(X_test_orig)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        spd = np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
              np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
        eo = np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
             np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
    elif cleaning == 'autosklearn':
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn_add_custom_clfs import add_clf

        if model_name == 'DT':
            clf_name = 'CustomDecisionTree'
        elif model_name == 'RF':
            clf_name = 'CustomRandomForest'
        elif model_name == 'SVM':
            clf_name = 'CustomSVM'
        elif model_name == 'NN':
            clf_name = 'CustomMLPClassifier'
        elif model_name == 'LR':
            clf_name = 'CustomLogisticRegression'
        else:
            raise ValueError("Unsupported autosklearn model name")

        add_clf(clf_name)
        overall_time = 30
        per_run_time = overall_time / 5
        clf = AutoSklearnClassifier(time_left_for_this_task=overall_time, per_run_time_limit=per_run_time,
                                    include={'classifier': [clf_name]}, memory_limit=1024 * 8)
        result = None
        try:
            clf.fit(dirty_X_train_orig, dirty_y_train)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            if y_pred_proba.shape[1] == 1 and error_type == 'Sampling':
                raise optuna.exceptions.TrialPruned()
            else:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            spd = np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
                  np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
            eo = np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
                 np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
        except ValueError as e:
            if 'Requesting 5-fold cross-validation but provided less than 5 examples for at least one class' in str(e):
                raise optuna.exceptions.TrialPruned()
            else:
                raise e
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise optuna.exceptions.TrialPruned()
    elif cleaning == 'random':
        from random_search import RandomSearch
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn_add_custom_clfs import add_clf

        if model_name == 'DT':
            clf_name = 'DecisionTree'
        elif model_name == 'RF':
            clf_name = 'RandomForest'
        elif model_name == 'SVM':
            clf_name = 'SVM'
        elif model_name == 'NN':
            clf_name = 'MLPClassifier'
        else:
            clf_name = 'LogisticRegression'

        X_train_orig, X_val_orig, y_train, y_val = \
            train_test_split(dirty_X_train_orig, dirty_y_train, test_size=0.25, random_state=42)
        X_train_orig, X_val_orig, X_test_orig = (X_train_orig.reset_index(drop=True),
                                                 X_val_orig.reset_index(drop=True),
                                                 X_test_orig.reset_index(drop=True))
        y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            raise optuna.exceptions.TrialPruned()

        fitted_clf, val_auc = RandomSearch(X_train_orig, y_train, X_val_orig, y_val,
                                           clf_name=clf_name, n_trials=10)

        y_pred = fitted_clf.predict(X_test)
        auc = roc_auc_score(y_test, fitted_clf.predict_proba(X_test_orig)[:, 1])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        spd = np.mean(fitted_clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
              np.mean(fitted_clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
        eo = np.mean(fitted_clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
             np.mean(fitted_clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])

    elif cleaning == 'diffprep':
        from diffprep.utils import SummaryWriter
        from diffprep.prep_space import space
        from diffprep.experiment.diffprep_experiment import DiffPrepExperiment
        from diffprep.pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
        from diffprep.trainer.diffprep_trainer import DiffPrepSGD
        from diffprep.model import LogisticRegression as DiffprepLogisticRegression
        from diffprep.experiment.experiment_utils import min_max_normalize, set_random_seed

        X_train, X_val, y_train, y_val = train_test_split(
            dirty_X_train_orig, dirty_y_train, test_size=0.5, random_state=42
        )
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        unique_train_labels = np.unique(y_train)
        unique_val_labels = np.unique(y_val)
        unique_test_labels = np.unique(y_test)

        if len(unique_train_labels) == 1 or len(unique_val_labels) == 1 or len(unique_test_labels) == 1:
            raise optuna.exceptions.TrialPruned()

        sensitive_attr_train = X_train[sens_attr].values
        sensitive_attr_val = X_val[sens_attr].values
        sensitive_attr_test = X_test[sens_attr].values if X_test is not None else None

        X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
        params["patience"] = 10
        params["num_epochs"] = 100
        set_random_seed(params)
        prep_pipeline = DiffPrepFixPipeline(space, temperature=params["temperature"],
                                            use_sample=params["sample"],
                                            diff_method=params["diff_method"],
                                            init_method=params["init_method"])
        prep_pipeline.init_parameters(X_train, X_val, X_test)
        print("Train size: ({}, {})".format(X_train.shape[0], prep_pipeline.out_features))

        # model
        input_dim = prep_pipeline.out_features
        output_dim = len(set(y_train.values.ravel()))

        # model = TwoLayerNet(input_dim, output_dim)
        set_random_seed(params)
        model = DiffprepLogisticRegression(input_dim, output_dim)
        model = model.to(params["device"])
        # loss
        loss_fn = nn.CrossEntropyLoss()

        # optimizer
        model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params["model_lr"],
            weight_decay=params["weight_decay"],
            momentum=params["momentum"]
        )

        if params["prep_lr"] is None:
            prep_lr = params["model_lr"]
        else:
            prep_lr = params["prep_lr"]

        prep_pipeline_optimizer = torch.optim.Adam(
            prep_pipeline.parameters(),
            lr=prep_lr,
            betas=(0.5, 0.999),
            weight_decay=params["weight_decay"]
        )

        # scheduler
        # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=patience, factor=0.1, threshold=0.001)
        prep_pipeline_scheduler = None
        model_scheduler = None

        if params["logging"]:
            logger = SummaryWriter()
        else:
            logger = None

        diff_prep = DiffPrepSGD(prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
                                model_scheduler, prep_pipeline_scheduler, params, writer=logger)

        result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, sensitive_attr_val, sensitive_attr_test,
                                           X_test,
                                           y_test)

        best_result = None
        best_model = None
        best_logger = None
        best_val_loss = float("inf")
        best_params = None

        if result["best_val_loss"] < best_val_loss:
            best_val_loss = result["best_val_loss"]
            best_result = result
            best_model = model
            best_logger = logger
            best_params = params

        auc = result["best_test_auc"]
        acc = result["best_test_acc"]
        f1 = f1_score(y_test, result["best_test_pred"])
        spd = result["best_spd"]
        eo = result["best_eo"]
    else:
        raise ValueError("Unsupported cleaning approach name")

    auc_drop = auc - baseline_auc

    #     print(time.time()-t0)
    #     t0 = time.time()
    #     print('-------------------------')

    trial.set_user_attr('num_errs', mv_num)
    trial.set_user_attr('auc', abs(auc))
    trial.set_user_attr('spd', abs(spd))
    trial.set_user_attr('eo', abs(eo))
    trial.set_user_attr('accuracy', abs(acc))
    trial.set_user_attr('f1_score', abs(f1))
    trial.set_user_attr('error_injector', injecter)
    trial.set_user_attr('col_list', current_pattern_col_list)
    trial.set_user_attr('lb_list', lb_list)
    trial.set_user_attr('ub_list', ub_list)

    if objective == 'SPD':
        return -abs(spd)
    elif objective == 'EO':
        return -abs(eo)
    else:
        return auc_drop
# def objective_fuc(trial, col_id, budget, baseline_auc, pattern_col_list, error_type, dataset, model_name, objective, sens_attr, cleaning):
#     lb_list = []
#     ub_list = []
#     X_train, X_test, y_train, y_test = load(dataset)
#     X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
#
#     X_train_orig.reset_index(drop=True, inplace=True)
#     X_test_orig.reset_index(drop=True, inplace=True)
#     y_train.reset_index(drop=True, inplace=True)
#     y_test.reset_index(drop=True, inplace=True)
#
#     if error_type == 'MAR':
#         pattern_col_list_copy = copy.deepcopy(pattern_col_list)
#         excluded_col = X_train_orig.columns[col_id]
#         if excluded_col in pattern_col_list_copy:
#             pattern_col_list_copy.remove(excluded_col)
#         current_pattern_col_list = pattern_col_list_copy
#     else:  # Default to MNAR if not specified
#         current_pattern_col_list = pattern_col_list
#     def map_percent_to_value(data, percent):
#         sorted_data = data.sort_values().reset_index(drop=True)
#         percent = min(max(percent, 0), 1)
#         index = int(percent * (len(sorted_data) - 1))
#         return sorted_data.iloc[index]
#
#     for col in current_pattern_col_list:
#         if col == 'Y':
#             if pd.api.types.is_integer_dtype(y_train):
#                 mv_lb = trial.suggest_int(col + '_mv_lb', int(y_train.min()), int(y_train.max()))
#                 mv_interval = trial.suggest_int(col + '_mv_int', 0, int(y_train.max()) - mv_lb)
#             else:
#                 mv_lb = trial.suggest_float(col + '_mv_lb', y_train.min(), y_train.max())
#                 mv_interval = trial.suggest_float(col + '_mv_int', 0, y_train.max() - mv_lb)
#             lb_list.append(mv_lb)
#             mv_ub = mv_interval + mv_lb
#             ub_list.append(mv_ub)
#         else:
#             if pd.api.types.is_integer_dtype(X_train_orig[col]):
#                 mv_lb = trial.suggest_int(col + '_mv_lb', int(X_train_orig[col].min()), int(X_train_orig[col].max()))
#                 mv_interval = trial.suggest_int(col + '_mv_int', 0, int(X_train_orig[col].max()) - mv_lb)
#                 mv_ub = mv_interval + mv_lb
#             else:
#                 print("test")
#                 mv_lb_percent = trial.suggest_float(col + '_mv_lb', 0, 1)
#                 mv_interval_percent = trial.suggest_float(col + '_mv_int', 0, 1 - mv_lb_percent)
#                 mv_ub_percent = mv_interval_percent + mv_lb_percent
#                 # map precent to real value
#                 print(
#                     f"Before mapping to actual values: mv_lb_percent: {mv_lb_percent}, mv_ub_percent: {mv_ub_percent}")
#                 mv_lb = map_percent_to_value(X_train_orig[col], mv_lb_percent)
#                 mv_ub = map_percent_to_value(X_train_orig[col], mv_ub_percent)
#                 print(f"After mapping to actual values: mv_lb: {mv_lb}, mv_ub: {mv_ub}")
#             lb_list.append(mv_lb)
#             ub_list.append(mv_ub)
#
#     mv_pattern = create_pattern(current_pattern_col_list, lb_list, ub_list)
#     mv_pattern_len = np.sum(mv_pattern(X_train_orig, y_train))
#
#     if mv_pattern_len == 0:
#         raise optuna.exceptions.TrialPruned()
#     mv_num = min(mv_pattern_len, budget)
#
#     if error_type == 'Label':
#         label_flip_ratio = min(mv_pattern_len, budget) / mv_pattern_len
#         mv_err = LabelError(pattern=mv_pattern, ratio=label_flip_ratio)
#     elif error_type == 'Sampling':
#         sampling_error_ratio = min(mv_pattern_len, budget) / mv_pattern_len
#         mv_err = SamplingError(pattern=mv_pattern, ratio=sampling_error_ratio)
#     else:
#         mv_err = MissingValueError(col_id, mv_pattern, mv_num / mv_pattern_len)
#
#     injecter = Injector(error_seq=[mv_err])
#     dirty_X_train_orig, dirty_y_train, _, _ = injecter.inject(X_train_orig.copy(), y_train.copy(), X_train_orig, y_train, seed=42)
#
#     if len(np.unique(dirty_y_train)) < 2:
#         raise optuna.exceptions.TrialPruned()
#
#     if cleaning == 'boostclean':
#
#         model = get_model(model_name)
#         acc, auc, spd, eo, f1 = boost_clean(model, dirty_X_train_orig,
#                                             dirty_y_train.to_numpy(),
#                                             X_test_orig, y_test.to_numpy(),
#                                             X_test_sensitive=X_test_orig[sens_attr].copy(), T=3)
#     elif cleaning == 'h2o':
#         try:
#             dirty_X_train_orig = SimpleImputer(strategy='mean').fit_transform(dirty_X_train_orig)
#             ss = StandardScaler()
#             ss.fit(dirty_X_train_orig)
#             X_train = ss.transform(dirty_X_train_orig)
#             X_test = ss.transform(X_test_orig)
#
#             unique_train_labels = np.unique(dirty_y_train)
#             unique_test_labels = np.unique(y_test)
#
#             if len(unique_train_labels) == 1 or len(unique_test_labels) == 1:
#                 raise optuna.exceptions.TrialPruned()
#
#             clf = get_model(model_name)
#             clf.fit(X_train, dirty_y_train)
#
#             y_pred = clf.predict(X_test)
#             y_pred_proba = clf.predict_proba(X_test)
#             if y_pred_proba.shape[1] == 1 and error_type == 'Sampling':
#                 raise optuna.exceptions.TrialPruned()
#             else:
#                 auc = roc_auc_score(y_test, y_pred_proba[:, 1])
#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
#             spd = np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
#                   np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
#             eo = np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
#                  np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
#         except ValueError as e:
#             if 'Requesting 5-fold cross-validation but provided less than 5 examples for at least one class' in str(e):
#                 raise optuna.exceptions.TrialPruned()
#             else:
#                 raise e
#     elif cleaning in ['MeanImputer', 'MedianImputer', 'KNNImputer', 'IterativeImputer']:
#         imputers = {
#             'MeanImputer': SimpleImputer(strategy='mean'),
#             'MedianImputer': SimpleImputer(strategy='median'),
#             'KNNImputer': KNNImputer(n_neighbors=10),
#             'IterativeImputer': IterativeImputer(max_iter=10, random_state=42)
#         }
#         imputer = imputers[cleaning]
#         dirty_X_train_imputed = pd.DataFrame(imputer.fit_transform(dirty_X_train_orig),
#                                              columns=dirty_X_train_orig.columns)
#
#         clf = get_model(model_name)
#         clf.fit(dirty_X_train_imputed, dirty_y_train)
#
#         y_pred = clf.predict(X_test_orig)
#         y_pred_proba = clf.predict_proba(X_test_orig)
#         auc = roc_auc_score(y_test, y_pred_proba[:, 1])
#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         spd = np.mean(clf.predict(X_test_orig)[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
#               np.mean(clf.predict(X_test_orig)[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
#         eo = np.mean(clf.predict(X_test_orig)[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
#              np.mean(clf.predict(X_test_orig)[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
#     elif cleaning == 'autosklearn':
#         from autosklearn.classification import AutoSklearnClassifier
#         from autosklearn_add_custom_clfs import add_clf
#
#         if model_name == 'DT':
#             clf_name = 'CustomDecisionTree'
#         elif model_name == 'RF':
#             clf_name = 'CustomRandomForest'
#         elif model_name == 'SVM':
#             clf_name = 'CustomSVM'
#         elif model_name == 'NN':
#             clf_name = 'CustomMLPClassifier'
#         elif model_name == 'LR':
#             clf_name = 'CustomLogisticRegression'
#         else:
#             raise ValueError("Unsupported autosklearn model name")
#
#         add_clf(clf_name)
#         overall_time = 30
#         per_run_time = overall_time / 5
#         clf = AutoSklearnClassifier(time_left_for_this_task=overall_time, per_run_time_limit=per_run_time,
#                                     include={'classifier': [clf_name]}, memory_limit=1024 * 8)
#         result = None
#         try:
#             clf.fit(dirty_X_train_orig, dirty_y_train)
#             y_pred = clf.predict(X_test)
#             y_pred_proba = clf.predict_proba(X_test)
#             if y_pred_proba.shape[1] == 1 and error_type == 'Sampling':
#                 raise optuna.exceptions.TrialPruned()
#             else:
#                 auc = roc_auc_score(y_test, y_pred_proba[:, 1])
#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
#             spd = np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
#                   np.mean(clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
#             eo = np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
#                  np.mean(clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
#         except ValueError as e:
#             if 'Requesting 5-fold cross-validation but provided less than 5 examples for at least one class' in str(e):
#                 raise optuna.exceptions.TrialPruned()
#             else:
#                 raise e
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#             raise optuna.exceptions.TrialPruned()
#     elif cleaning == 'random':
#         from random_search import RandomSearch
#         from autosklearn.classification import AutoSklearnClassifier
#         from autosklearn_add_custom_clfs import add_clf
#
#         if model_name == 'DT':
#             clf_name = 'DecisionTree'
#         elif model_name == 'RF':
#             clf_name = 'RandomForest'
#         elif model_name == 'SVM':
#             clf_name = 'SVM'
#         elif model_name == 'NN':
#             clf_name = 'MLPClassifier'
#         else:
#             clf_name = 'LogisticRegression'
#
#         X_train_orig, X_val_orig, y_train, y_val = \
#             train_test_split(dirty_X_train_orig, dirty_y_train, test_size=0.25, random_state=42)
#         X_train_orig, X_val_orig, X_test_orig = (X_train_orig.reset_index(drop=True),
#                                                  X_val_orig.reset_index(drop=True),
#                                                  X_test_orig.reset_index(drop=True))
#         y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)
#
#         if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
#             raise optuna.exceptions.TrialPruned()
#
#         fitted_clf, val_auc = RandomSearch(X_train_orig, y_train, X_val_orig, y_val,
#                                            clf_name=clf_name, n_trials=10)
#
#         y_pred = fitted_clf.predict(X_test)
#         auc = roc_auc_score(y_test, fitted_clf.predict_proba(X_test_orig)[:, 1])
#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         spd = np.mean(fitted_clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
#               np.mean(fitted_clf.predict(X_test)[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
#         eo = np.mean(fitted_clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
#              np.mean(fitted_clf.predict(X_test)[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
#
#     elif cleaning == 'diffprep':
#         from diffprep.utils import SummaryWriter
#         from diffprep.prep_space import space
#         from diffprep.experiment.diffprep_experiment import DiffPrepExperiment
#         from diffprep.pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
#         from diffprep.trainer.diffprep_trainer import DiffPrepSGD
#         from diffprep.model import LogisticRegression as DiffprepLogisticRegression
#         from diffprep.experiment.experiment_utils import min_max_normalize, set_random_seed
#
#         X_train, X_val, y_train, y_val = train_test_split(
#             dirty_X_train_orig, dirty_y_train, test_size=0.5, random_state=42
#         )
#         X_train = X_train.reset_index(drop=True)
#         X_val = X_val.reset_index(drop=True)
#         y_train = y_train.reset_index(drop=True)
#         y_val = y_val.reset_index(drop=True)
#         X_test = X_test.reset_index(drop=True)
#
#         unique_train_labels = np.unique(y_train)
#         unique_val_labels = np.unique(y_val)
#         unique_test_labels = np.unique(y_test)
#
#         if len(unique_train_labels) == 1 or len(unique_val_labels) == 1 or len(unique_test_labels) == 1:
#             raise optuna.exceptions.TrialPruned()
#
#         sensitive_attr_train = X_train[sens_attr].values
#         sensitive_attr_val = X_val[sens_attr].values
#         sensitive_attr_test = X_test[sens_attr].values if X_test is not None else None
#
#         X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
#         params["patience"] = 10
#         params["num_epochs"] = 100
#         set_random_seed(params)
#         prep_pipeline = DiffPrepFixPipeline(space, temperature=params["temperature"],
#                                             use_sample=params["sample"],
#                                             diff_method=params["diff_method"],
#                                             init_method=params["init_method"])
#         prep_pipeline.init_parameters(X_train, X_val, X_test)
#         print("Train size: ({}, {})".format(X_train.shape[0], prep_pipeline.out_features))
#
#         # model
#         input_dim = prep_pipeline.out_features
#         output_dim = len(set(y_train.values.ravel()))
#
#         # model = TwoLayerNet(input_dim, output_dim)
#         set_random_seed(params)
#         model = DiffprepLogisticRegression(input_dim, output_dim)
#         model = model.to(params["device"])
#         # loss
#         loss_fn = nn.CrossEntropyLoss()
#
#         # optimizer
#         model_optimizer = torch.optim.SGD(
#             model.parameters(),
#             lr=params["model_lr"],
#             weight_decay=params["weight_decay"],
#             momentum=params["momentum"]
#         )
#
#         if params["prep_lr"] is None:
#             prep_lr = params["model_lr"]
#         else:
#             prep_lr = params["prep_lr"]
#
#         prep_pipeline_optimizer = torch.optim.Adam(
#             prep_pipeline.parameters(),
#             lr=prep_lr,
#             betas=(0.5, 0.999),
#             weight_decay=params["weight_decay"]
#         )
#
#         # scheduler
#         # model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, patience=patience, factor=0.1, threshold=0.001)
#         prep_pipeline_scheduler = None
#         model_scheduler = None
#
#         if params["logging"]:
#             logger = SummaryWriter()
#         else:
#             logger = None
#
#         diff_prep = DiffPrepSGD(prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
#                                 model_scheduler, prep_pipeline_scheduler, params, writer=logger)
#
#         result, best_model = diff_prep.fit(X_train, y_train, X_val, y_val, sensitive_attr_val, sensitive_attr_test,
#                                            X_test,
#                                            y_test)
#
#         best_result = None
#         best_model = None
#         best_logger = None
#         best_val_loss = float("inf")
#         best_params = None
#
#         if result["best_val_loss"] < best_val_loss:
#             best_val_loss = result["best_val_loss"]
#             best_result = result
#             best_model = model
#             best_logger = logger
#             best_params = params
#
#         auc = result["best_test_auc"]
#         acc = result["best_test_acc"]
#         f1 = f1_score(y_test, result["best_test_pred"])
#         spd = result["best_spd"]
#         eo = result["best_eo"]
#     else:
#         raise ValueError("Unsupported cleaning approach name")
#
#     auc_drop = auc - baseline_auc
#
#     trial.set_user_attr('num_errs', mv_num)
#     trial.set_user_attr('auc', abs(auc))
#     trial.set_user_attr('spd', abs(spd))
#     trial.set_user_attr('eo', abs(eo))
#     trial.set_user_attr('accuracy', abs(acc))
#     trial.set_user_attr('f1_score', abs(f1))
#     trial.set_user_attr('error_injector', injecter)
#     trial.set_user_attr('col_list', current_pattern_col_list)
#     trial.set_user_attr('lb_list', lb_list)
#     trial.set_user_attr('ub_list', ub_list)
#
#     if objective == 'SPD':
#         return -abs(spd)
#     elif objective == 'EO':
#         return -abs(eo)
#     else:
#         return auc_drop

def run_one_setting(setting_dict):
    dataset = setting_dict['dataset']
    model_name = setting_dict['model_name']
    pattern_col_list = setting_dict['pattern_col_list']
    objective = setting_dict['objective']
    sens_attr = setting_dict['sens_attr']
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
    baseline_auc = setting_dict['baseline_auc']
    seed = setting_dict['seed']
    missing_col = setting_dict['missing_col']
    budget = setting_dict['budget']
    error_type = setting_dict['error_type']
    n_trials = setting_dict['n_trials']
    cleaning = setting_dict['cleaning']

    col_id = list(X_train_orig.columns).index(missing_col)
    print(f"Evaluating seed: {seed} for {missing_col} with {budget} NaNs...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    study.optimize(
        lambda trial: objective_fuc(trial, col_id, budget, baseline_auc, pattern_col_list, error_type, dataset,
                                    model_name,
                                    objective, sens_attr, cleaning),
        n_trials=n_trials)

    auc = study.best_trial.user_attrs['auc']
    spd = study.best_trial.user_attrs['spd']
    eo = study.best_trial.user_attrs['eo']
    acc = study.best_trial.user_attrs['accuracy']
    f1 = study.best_trial.user_attrs['f1_score']
    col_list = study.best_trial.user_attrs['col_list']
    lb_list = study.best_trial.user_attrs['lb_list']
    ub_list = study.best_trial.user_attrs['ub_list']

    return seed, missing_col, budget, auc, spd, eo, acc, f1, col_list, lb_list, ub_list


def move_zero_to_front(data):
    for metric in ["AUC", "SPD", "EO", "ACC", "F1"]:
        if 0 in data[metric]['budgets']:
            idx_0 = data[metric]['budgets'].index(0)
            if idx_0 != 0:
                data[metric]['budgets'].insert(0, data[metric]['budgets'].pop(idx_0))
                data[metric]['means'].insert(0, data[metric]['means'].pop(idx_0))
                data[metric]['stds'].insert(0, data[metric]['stds'].pop(idx_0))

    for key in ["col_list", "lb_list", "ub_list"]:
        if data[key]:
            for i in range(len(data[key])):
                for j in range(len(data[key][i])):
                    if data[key][i][j] == 0:
                        data[key][i].insert(0, data[key][i].pop(j))

def merge_results(final_results, result_filename):
    with open(result_filename, 'r') as json_file:
        existing_results = json.load(json_file)

    for col in final_results:
        move_zero_to_front(final_results[col])
        for metric in ['AUC', 'SPD', 'EO', 'ACC', 'F1']:
            existing_results[col][metric]['budgets'].extend(final_results[col][metric]['budgets'])
            existing_results[col][metric]['means'].extend(final_results[col][metric]['means'])
            existing_results[col][metric]['stds'].extend(final_results[col][metric]['stds'])

        for key in ['col_list', 'lb_list', 'ub_list']:
            existing_results[col][key].extend(final_results[col][key])

    for col in existing_results:
        move_zero_to_front(existing_results[col])

    return existing_results

def main():
    parser = argparse.ArgumentParser(description="Run data integrity attacks on classification models.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--cleaning', type=str, required=True, help='Cleaning approach/framework')
    parser.add_argument('--model', type=str, required=True, choices=['LR', 'DT', 'NN', 'RF', 'SVM'], help='Downstream model name')
    parser.add_argument('--error_type', type=str, required=True, choices=['MNAR', 'MAR', 'Sampling', 'Label'], help='Error type')
    parser.add_argument('--error_pct', type=float, default=0.1, help='Error number percentage (default: 10%)')
    parser.add_argument('--error_cols', type=str, nargs='+', required=True, help='Error injection columns')
    parser.add_argument('--pattern_cols', type=str, nargs='+', required=True, help='Pattern selection columns')
    parser.add_argument('--sens_attr', type=str, required=True, help='Sensitive attribute')
    parser.add_argument('--objective', type=str, required=True, choices=['AUC', 'EO', 'SPD'], help='Attack objective')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna optimization')
    parser.add_argument('--n_processes', type=int, default=10, help='Number of processes for multiprocessing (default: 10)')
    parser.add_argument('--override', action='store_true', help='Override existing results')

    args = parser.parse_args()

    if not os.path.exists('./save/'):
        os.system('mkdir ./save/')

    global pattern_col_list, n_trials, processes
    dataset = args.dataset
    cleaning = args.cleaning
    model_name = args.model
    error_type = args.error_type
    error_pct = args.error_pct
    error_cols = args.error_cols
    pattern_cols = args.pattern_cols
    sens_attr = args.sens_attr
    attack_objective = args.objective
    n_trials = args.n_trials
    processes = args.n_processes
    override = args.override

    pattern_col_list = pattern_cols
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
    X_test_orig.reset_index(drop=True, inplace=True)
    nan_counts = [0] if not override else [0] + [int(pct * len(X_train_orig)) for pct in np.arange(error_pct, error_pct * 5 + error_pct, error_pct)]
    seed_values = [42, 43, 44, 45, 46]

    baseline_auc = initialize_model_and_baseline_auc(dataset, model_name, cleaning, sens_attr)

    settings_list = [
        {'dataset': dataset, 'model_name': model_name, 'error_type': error_type, 'baseline_auc': baseline_auc,
         'seed': seed,
         'missing_col': missing_col, 'budget': budget, 'n_trials': n_trials, 'pattern_col_list': pattern_cols,
         'objective': attack_objective, 'sens_attr': sens_attr, 'cleaning': cleaning}
        for seed in seed_values for missing_col in error_cols for budget in nan_counts]

    final_results = {
        col: {
            'AUC': {'means': [], 'stds': [], 'budgets': []},
            'SPD': {'means': [], 'stds': [], 'budgets': []},
            'EO': {'means': [], 'stds': [], 'budgets': []},
            'ACC': {'means': [], 'stds': [], 'budgets': []},
            'F1': {'means': [], 'stds': [], 'budgets': []},
            'col_list': {budget: [] for budget in nan_counts},
            'lb_list': {budget: [] for budget in nan_counts},
            'ub_list': {budget: [] for budget in nan_counts}
        } for col in error_cols
    }

    if processes == 1:
        results = [run_one_setting(setting) for setting in settings_list]
    else:
        with Pool(processes=processes) as pool:
            results = pool.map(run_one_setting, settings_list)

    for result in results:
        seed, col, budget, auc, spd, eo, acc, f1, col_list, lb_list, ub_list = result

        metrics = {
            'AUC': auc,
            'SPD': spd,
            'EO': eo,
            'ACC': acc,
            'F1': f1
        }

        for metric_name, value in metrics.items():
            if budget not in final_results[col][metric_name]['budgets']:
                final_results[col][metric_name]['budgets'].append(budget)
                final_results[col][metric_name]['means'].append([])
                final_results[col][metric_name]['stds'].append([])

            index = final_results[col][metric_name]['budgets'].index(budget)
            final_results[col][metric_name]['means'][index].append(value)

        final_results[col]['col_list'][budget].append(col_list)
        final_results[col]['lb_list'][budget].append(lb_list)
        final_results[col]['ub_list'][budget].append(ub_list)

    for col in final_results:
        for metric in ['AUC', 'SPD', 'EO', 'ACC', 'F1']:
            budgets = final_results[col][metric]['budgets']
            for i, budget in enumerate(budgets):
                scores = final_results[col][metric]['means'][i]
                final_results[col][metric]['means'][i] = np.mean(scores)
                final_results[col][metric]['stds'][i] = np.std(scores)

        for key in ['col_list', 'lb_list', 'ub_list']:
            sorted_budgets = sorted(final_results[col][key].keys())
            final_results[col][key] = [final_results[col][key][budget] for budget in sorted_budgets]

    if error_type in ['Label', 'Sampling']:
        final_results = {'full': final_results[error_cols[0]]}

    result_filename = f"{dataset}_results_{cleaning}_{error_type}_{attack_objective}_{model_name}.json"
    if not override and os.path.exists(result_filename):
        final_results = merge_results(final_results, result_filename)

    with open(result_filename, 'w') as json_file:
        json.dump(final_results, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()