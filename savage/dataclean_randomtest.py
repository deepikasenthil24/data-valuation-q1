import copy
# import tqdm
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

# Additional parameters for diffprep (you can adjust as needed)
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
    "patience": 10,
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
    X_train_orig = copy.deepcopy(X_train)
    X_test_orig = copy.deepcopy(X_test)
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(
        X_train_orig, y_train, test_size=0.25, random_state=42)

    X_train_orig, X_val_orig, X_test_orig = (
        X_train_orig.reset_index(drop=True),
        X_val_orig.reset_index(drop=True),
        X_test_orig.reset_index(drop=True))
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    if cleaning == 'boostclean':
        model = get_model(model_name)
        _, baseline_auc, _, _, _ = boost_clean(
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
        per_run_time = int(overall_time / 5)
        clf = AutoSklearnClassifier(
            time_left_for_this_task=overall_time,
            per_run_time_limit=per_run_time,
            include={'classifier': [clf_name]},
            memory_limit=1024 * 8
        )
        clf.fit(X_train_orig, y_train)
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

        fitted_clf, val_auc = RandomSearch(
            X_train_orig, y_train, X_val_orig, y_val,
            clf_name=clf_name, n_trials=10)
        baseline_auc = roc_auc_score(y_test, fitted_clf.predict_proba(X_test_orig)[:, 1])
    elif cleaning == 'diffprep':
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_orig, y_train, test_size=0.5, random_state=42)
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        X_test = X_test_orig.reset_index(drop=True)

        unique_train_labels = np.unique(y_train)
        unique_val_labels = np.unique(y_val)
        unique_test_labels = np.unique(y_test)

        if len(unique_train_labels) == 1 or len(unique_val_labels) == 1 or len(unique_test_labels) == 1:
            raise ValueError("Not enough classes for training.")

        sensitive_attr_train = X_train[sens_attr].values
        sensitive_attr_val = X_val[sens_attr].values
        sensitive_attr_test = X_test[sens_attr].values if X_test is not None else None

        X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
        set_random_seed(params)
        prep_pipeline = DiffPrepFixPipeline(
            space, temperature=params["temperature"],
            use_sample=params["sample"],
            diff_method=params["diff_method"],
            init_method=params["init_method"]
        )
        prep_pipeline.init_parameters(X_train, X_val, X_test)

        input_dim = prep_pipeline.out_features
        output_dim = len(set(y_train.values.ravel()))

        set_random_seed(params)
        model = DiffprepLogisticRegression(input_dim, output_dim)
        model = model.to(params["device"])
        loss_fn = nn.CrossEntropyLoss()

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

        model_scheduler = None
        prep_pipeline_scheduler = None

        logger = SummaryWriter() if params["logging"] else None

        diff_prep = DiffPrepSGD(
            prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
            model_scheduler, prep_pipeline_scheduler, params, writer=logger
        )

        result, _ = diff_prep.fit(
            X_train, y_train, X_val, y_val, sensitive_attr_val, sensitive_attr_test, X_test, y_test
        )

        baseline_auc = result["best_test_auc"]
    else:
        raise ValueError("Unsupported cleaning approach name")

    return baseline_auc

def get_model(model_name, seed=42):
    if model_name == 'LR':
        return LogisticRegression(random_state=seed, max_iter=1000)
    elif model_name == 'DT':
        return DecisionTreeClassifier(random_state=seed)
    elif model_name == 'NN':
        return MLPClassifier(random_state=seed, hidden_layer_sizes=(10,))
    elif model_name == 'RF':
        return RandomForestClassifier(random_state=seed)
    elif model_name == 'SVM':
        sgd_clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=1000, tol=1e-3)
        return CalibratedClassifierCV(sgd_clf, method='sigmoid', cv=5)
    else:
        raise ValueError("Unsupported model name")

def create_pattern(col_list, lb_list, ub_list):
    # Check if inputs are valid
    try:
        assert len(col_list) == len(lb_list) == len(ub_list)
    except:
        print(col_list, lb_list, ub_list)
        raise SyntaxError

    def pattern(data_X, data_y):
        # Initialize a mask of all True values
        mask = np.ones(len(data_X), dtype=bool)

        # Iterate over each condition in col_list, lb_list, and ub_list
        for col, lb, ub in zip(col_list, lb_list, ub_list):
            if col == 'Y':
                mask &= (data_y >= lb) & (data_y <= ub)
            else:
                mask &= (data_X[col] >= lb) & (data_X[col] <= ub)

        # Convert Boolean mask to binary indicators (1 for True, 0 for False)
        binary_indicators = mask.astype(int)
        return binary_indicators

    return pattern

def map_percent_to_value(data, percent):
    sorted_data = data.sort_values().reset_index(drop=True)
    percent = min(max(percent, 0), 1)
    index = int(percent * (len(sorted_data) - 1))
    return sorted_data.iloc[index]

def run_random_test(dataset, model_name, cleaning, sens_attr, columns, baseline_auc, seed):
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig = X_train.reset_index(drop=True)
    X_test_orig = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Randomly select an error_col (cannot be 'Y')
    error_cols = [col for col in columns if col != 'Y']
    error_col = random.choice(error_cols)

    # Randomly select 0-4 columns as pattern_cols
    pattern_col_num = random.randint(0, 4)
    pattern_cols = random.sample(columns, pattern_col_num)

    # Randomly select mv_lb and mv_ub
    lb_list = []
    ub_list = []

    for col in pattern_cols:
        col_data = X_train_orig[col]
        if pd.api.types.is_numeric_dtype(col_data):
            mv_lb_percent = random.uniform(0, 1)
            mv_ub_percent = random.uniform(mv_lb_percent, 1)
            mv_lb = map_percent_to_value(col_data, mv_lb_percent)
            mv_ub = map_percent_to_value(col_data, mv_ub_percent)
        else:
            unique_values = col_data.unique()
            mv_lb = random.choice(unique_values)
            mv_ub = mv_lb
        lb_list.append(mv_lb)
        ub_list.append(mv_ub)

    # Create missing value pattern
    mv_pattern = create_pattern(pattern_cols, lb_list, ub_list)
    mv_pattern_len = np.sum(mv_pattern(X_train_orig, y_train))

    if mv_pattern_len == 0:
        return None  # Skip if no data matches the pattern

    # Set missing value count (50% of the dataset)
    mv_num = int(0.5 * len(X_train_orig))
    mv_num = min(mv_pattern_len, mv_num)

    # Create missing value error
    col_id = list(X_train_orig.columns).index(error_col)
    mv_err = MissingValueError(col_id, mv_pattern, mv_num / mv_pattern_len)

    # Inject errors
    injecter = Injector(error_seq=[mv_err])
    dirty_X_train_orig, dirty_y_train, _, _ = injecter.inject(
        X_train_orig.copy(), y_train.copy(), X_train_orig, y_train, seed=seed
    )

    # Data cleaning based on the selected method
    if cleaning == 'boostclean':
        model = get_model(model_name, seed)
        _, auc, _, _, _ = boost_clean(model, dirty_X_train_orig,
                                      dirty_y_train.to_numpy(), X_test_orig,
                                      y_test.to_numpy(),
                                      X_test_sensitive=X_test_orig[sens_attr].copy(), T=3)
    elif cleaning == 'h2o':
        dirty_X_train_imputed = SimpleImputer(strategy='mean').fit_transform(dirty_X_train_orig)
        ss = StandardScaler()
        ss.fit(dirty_X_train_imputed)
        X_train = ss.transform(dirty_X_train_imputed)
        X_test = ss.transform(X_test_orig)

        clf = get_model(model_name, seed)
        clf.fit(X_train, dirty_y_train)

        y_test_pred = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_test_pred)
    elif cleaning in ['MeanImputer', 'MedianImputer', 'KNNImputer', 'IterativeImputer']:
        imputers = {
            'MeanImputer': SimpleImputer(strategy='mean'),
            'MedianImputer': SimpleImputer(strategy='median'),
            'KNNImputer': KNNImputer(n_neighbors=10),
            'IterativeImputer': IterativeImputer(max_iter=10, random_state=seed)
        }
        imputer = imputers[cleaning]
        dirty_X_train_imputed = pd.DataFrame(imputer.fit_transform(dirty_X_train_orig),
                                             columns=dirty_X_train_orig.columns)
        clf = get_model(model_name, seed)
        clf.fit(dirty_X_train_imputed, dirty_y_train)
        y_test_pred_proba = clf.predict_proba(X_test_orig)[:, 1]
        auc = roc_auc_score(y_test, y_test_pred_proba)
    elif cleaning == 'autosklearn':
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
        per_run_time = int(overall_time / 5)
        clf = AutoSklearnClassifier(
            time_left_for_this_task=overall_time,
            per_run_time_limit=per_run_time,
            include={'classifier': [clf_name]},
            memory_limit=1024 * 8
        )
        clf.fit(dirty_X_train_orig, dirty_y_train)
        y_test_pred_proba = clf.predict_proba(X_test_orig)[:, 1]
        auc = roc_auc_score(y_test, y_test_pred_proba)
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

        X_train_orig, X_val_orig, y_train_split, y_val = train_test_split(
            dirty_X_train_orig, dirty_y_train, test_size=0.25, random_state=seed)
        X_train_orig = X_train_orig.reset_index(drop=True)
        X_val_orig = X_val_orig.reset_index(drop=True)
        y_train_split = y_train_split.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        if len(np.unique(y_train_split)) < 2 or len(np.unique(y_val)) < 2:
            return None

        fitted_clf, val_auc = RandomSearch(
            X_train_orig, y_train_split, X_val_orig, y_val,
            clf_name=clf_name, n_trials=10)

        y_test_pred_proba = fitted_clf.predict_proba(X_test_orig)[:, 1]
        auc = roc_auc_score(y_test, y_test_pred_proba)
    elif cleaning == 'diffprep':
        X_train, X_val, y_train_split, y_val = train_test_split(
            dirty_X_train_orig, dirty_y_train, test_size=0.5, random_state=seed
        )
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_train_split = y_train_split.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        X_test = X_test_orig.reset_index(drop=True)

        unique_train_labels = np.unique(y_train_split)
        unique_val_labels = np.unique(y_val)
        unique_test_labels = np.unique(y_test)

        if len(unique_train_labels) == 1 or len(unique_val_labels) == 1 or len(unique_test_labels) == 1:
            return None  # Skip if not enough classes

        sensitive_attr_train = X_train[sens_attr].values
        sensitive_attr_val = X_val[sens_attr].values
        sensitive_attr_test = X_test[sens_attr].values if X_test is not None else None

        X_train, X_val, X_test = min_max_normalize(X_train, X_val, X_test)
        set_random_seed(params)
        prep_pipeline = DiffPrepFixPipeline(
            space, temperature=params["temperature"],
            use_sample=params["sample"],
            diff_method=params["diff_method"],
            init_method=params["init_method"]
        )
        prep_pipeline.init_parameters(X_train, X_val, X_test)

        input_dim = prep_pipeline.out_features
        output_dim = len(set(y_train_split.values.ravel()))

        set_random_seed(params)
        model = DiffprepLogisticRegression(input_dim, output_dim)
        model = model.to(params["device"])
        loss_fn = nn.CrossEntropyLoss()

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

        model_scheduler = None
        prep_pipeline_scheduler = None

        logger = SummaryWriter() if params["logging"] else None

        diff_prep = DiffPrepSGD(
            prep_pipeline, model, loss_fn, model_optimizer, prep_pipeline_optimizer,
            model_scheduler, prep_pipeline_scheduler, params, writer=logger
        )

        result, _ = diff_prep.fit(
            X_train, y_train_split, X_val, y_val, sensitive_attr_val, sensitive_attr_test, X_test, y_test
        )

        auc = result["best_test_auc"]
    else:
        raise ValueError("Unsupported cleaning approach name")

    result = {
        'seed': seed,
        'error_col': error_col,
        'pattern_cols': pattern_cols,
        'lb_list': lb_list,
        'ub_list': ub_list,
        'auc': auc
    }

    return result

def main():
    dataset = 'employee'
    # cleaning_methods = ['h2o', 'diffprep', 'autosklearn', 'boostclean', 'MeanImputer', 'MedianImputer', 'KNNImputer',
    #                     'IterativeImputer', 'random']
    cleaning_methods = ['random']
    model_name = 'LR'
    sens_attr = 'Gender'
    columns = [
        "Education", "JoiningYear", "PaymentTier", "Age", "Gender", "EverBenched",
        "ExperienceInCurrentDomain", "City_Bangalore", "City_New Delhi", "City_Pune"
    ]

    for cleaning in cleaning_methods:
        print(f"\nRunning tests for cleaning method: {cleaning}")
        # Initialize baseline AUC
        baseline_auc = initialize_model_and_baseline_auc(dataset, model_name, cleaning, sens_attr)
        print(f"Baseline AUC for {cleaning}: {baseline_auc}")

        seed_values = [42, 43, 44, 45, 46]
        min_results_per_seed = []
        auc_values_per_seed = []

        for seed in seed_values:
            seed_results = []
            for i in range(100):
                result = run_random_test(
                    dataset, model_name, cleaning, sens_attr, columns, baseline_auc, seed
                )
                if result:
                    seed_results.append(result)

            if seed_results:
                min_result = min(seed_results, key=lambda x: x['auc'])
                auc_values_per_seed.append(min_result['auc'])
                min_results_per_seed.append(min_result)
                print(f"\nSeed {seed} minimum AUC result for {cleaning}:")
                print(f"Seed: {min_result['seed']}")
                print(f"Error Column: {min_result['error_col']}")
                print(f"Pattern Columns: {min_result['pattern_cols']}")
                print(f"Lower Bounds: {min_result['lb_list']}")
                print(f"Upper Bounds: {min_result['ub_list']}")
                print(f"AUC after error injection: {min_result['auc']}")
            else:
                print(f"No valid results found for seed {seed}")

        if auc_values_per_seed:
            mean_auc = np.mean(auc_values_per_seed)
            std_auc = np.std(auc_values_per_seed)
            print(f"\nMean of minimum AUCs across seeds for {cleaning}: {mean_auc}")
            print(f"Standard deviation of minimum AUCs: {std_auc}")
        else:
            print("No AUC values to calculate mean and standard deviation.")

        final_results = {
            'min_results_per_seed': min_results_per_seed,
            'mean_min_auc': mean_auc if auc_values_per_seed else None,
            'std_min_auc': std_auc if auc_values_per_seed else None
        }

        final_json_filename = f'min_auc_results_{cleaning}.json'
        with open(final_json_filename, 'w') as json_file:
            json.dump(final_results, json_file, indent=4, default=str)

if __name__ == '__main__':
    main()