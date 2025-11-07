import copy
import tqdm
import optuna
import pickle
import random_employee
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
from API_Design_a import MissingValueError, LabelError, SamplingError, Injector

import argparse
import os
import json
if not(os.path.exists('./save/')):
    os.system('mkdir ./save/')

dataset = 'google'

X_train, X_test, y_train, y_test = load(dataset)
X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
X_test_orig.reset_index(drop=True, inplace=True)

pattern_col_list = ['Size', 'Reviews', 'Install', 'Y']
# missing_col_list = ['Size', 'Reviews', 'Install']
percentages = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
nan_counts = [int(perc * len(X_train_orig)) for perc in percentages]
seed_values = [42, 43, 44, 45, 46]
n_trials = 100
processes = 10

def initialize_h2o_and_baseline_auc():
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig = copy.deepcopy(X_train)
    X_test_orig = copy.deepcopy(X_test)
    ss = StandardScaler()
    ss.fit(X_train_orig)
    X_train = ss.transform(X_train_orig)
    X_test = ss.transform(X_test_orig)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # predict using the model and the testing dataset
    y_test_pred = clf.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, y_test_pred)
    return baseline_auc

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
def objective(trial, col_id, budget, baseline_auc, pattern_col_list, type):
    lb_list = []
    ub_list = []
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
    X_test_orig.reset_index(drop=True, inplace=True)

    if type == 'MAR':
        pattern_col_list_copy = copy.deepcopy(pattern_col_list)
        excluded_col = X_train_orig.columns[col_id]
        if excluded_col in pattern_col_list_copy:
            pattern_col_list_copy.remove(excluded_col)
        current_pattern_col_list = pattern_col_list_copy
    else:  # Default to MNAR if not specified
        current_pattern_col_list = pattern_col_list

    for col in current_pattern_col_list:
        if col == 'Y':
            mv_lb = trial.suggest_int(col + '_mv_lb', y_train.min(), y_train.max())
            lb_list.append(mv_lb)
            mv_interval = trial.suggest_int(col + '_mv_int', 0, y_train.max() - mv_lb)
            mv_ub = mv_interval + mv_lb
            ub_list.append(mv_ub)
        else:
            mv_lb = trial.suggest_int(col + '_mv_lb', X_train_orig[col].min(), X_train_orig[col].max())
            lb_list.append(mv_lb)
            mv_interval = trial.suggest_int(col + '_mv_int', 0, X_train_orig[col].max() - mv_lb)
            mv_ub = mv_interval + mv_lb
            ub_list.append(mv_ub)

    mv_pattern = create_pattern(current_pattern_col_list, lb_list, ub_list)
    mv_pattern_len = np.sum(mv_pattern(X_train_orig, y_train))
    if mv_pattern_len == 0:
        # print("Pruning due to no matching pattern")
        raise optuna.exceptions.TrialPruned()
    mv_num = min(mv_pattern_len, budget)

    if type == 'Label':
        label_flip_ratio = min(mv_pattern_len, budget) / mv_pattern_len
        mv_err = LabelError(pattern=mv_pattern, ratio=label_flip_ratio)
    elif type == 'Sampling':
        sampling_error_ratio = min(mv_pattern_len, budget) / mv_pattern_len
        mv_err = SamplingError(pattern=mv_pattern, ratio=sampling_error_ratio)
    else:
        mv_err = MissingValueError(col_id, mv_pattern, mv_num / mv_pattern_len)

    injecter = Injector(error_seq=[mv_err])
    dirty_X_train_orig, dirty_y_train, _, _ = injecter.inject(X_train_orig.copy(), y_train.copy(),
                                                              X_train_orig, y_train, seed=42)

    if type == 'Label' and len(np.unique(dirty_y_train)) < 2:
        raise optuna.exceptions.TrialPruned()

    ss = StandardScaler()
    ss.fit(dirty_X_train_orig)
    X_train = ss.transform(dirty_X_train_orig)
    X_test = ss.transform(X_test_orig)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, dirty_y_train)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    if y_pred_proba.shape[1] == 1 and type == 'Sampling':
        raise optuna.exceptions.TrialPruned()
    else:
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    trial.set_user_attr('num_errs', mv_num)
    trial.set_user_attr('auc', abs(auc))
    trial.set_user_attr('accuracy', abs(acc))
    trial.set_user_attr('f1_score', abs(f1))
    trial.set_user_attr('error_injector', injecter)
    trial.set_user_attr('col_list', current_pattern_col_list)
    trial.set_user_attr('lb_list', lb_list)
    trial.set_user_attr('ub_list', ub_list)

    return auc

def run_one_setting(setting_dict):
    global final_results
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
    baseline_auc = setting_dict['baseline_auc']
    seed = setting_dict['seed']
    missing_col = setting_dict['missing_col']
    budget = setting_dict['budget']
    type = setting_dict['type']

    col_id = list(X_train_orig.columns).index(missing_col)
    print(f"Evaluating seed: {seed} for {missing_col} with {budget} NaNs...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    study.optimize(lambda trial: objective(trial, col_id, budget, baseline_auc, pattern_col_list, type), n_trials=n_trials)

    auc = study.best_trial.user_attrs['auc']
    acc = study.best_trial.user_attrs['accuracy']
    f1 = study.best_trial.user_attrs['f1_score']
    col_list = study.best_trial.user_attrs['col_list']
    lb_list = study.best_trial.user_attrs['lb_list']
    ub_list = study.best_trial.user_attrs['ub_list']

    return seed, missing_col, budget, auc, 0, 0, acc, f1, col_list, lb_list, ub_list

def main():
    # parser = argparse.ArgumentParser(description="Run data integrity attacks on classification models.")
    # parser.add_argument('--type', type=str, required=True, choices=['MNAR', 'MAR', 'SPD', 'EO', 'Sampling', 'Label'],
    #                     help='Type to simulate.')
    # args = parser.parse_args()
    type = 'Label'

    if type in ['Label', 'Sampling']:
        missing_col_list = ['Size']
    else:
        missing_col_list = ['Size', 'Reviews', 'Install']

    baseline_auc = initialize_h2o_and_baseline_auc()

    settings_list = [{'type': type,'baseline_auc': baseline_auc, 'seed': seed, 'missing_col': missing_col, 'budget': budget}
                     for seed in seed_values for missing_col in missing_col_list for budget in nan_counts]

    # Initialize the multiprocessing Pool
    with Pool(processes=processes) as pool:
        results = pool.map(run_one_setting, settings_list)

    # Initialize final_results with the necessary structure
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
        } for col in missing_col_list
    }

    # Process results and fill final_results
    for result in results:
        seed, col, budget, auc, spd, eo, acc, f1, col_list, lb_list, ub_list = result

        # Collect results for all metrics
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

        # Append the lists for each budget and seed
        final_results[col]['col_list'][budget].append(col_list)
        final_results[col]['lb_list'][budget].append(lb_list)
        final_results[col]['ub_list'][budget].append(ub_list)

    # Calculate means and stds for each budget and metric
    for col in final_results:
        for metric in ['AUC', 'SPD', 'EO', 'ACC', 'F1']:
            budgets = final_results[col][metric]['budgets']
            for i, budget in enumerate(budgets):
                scores = final_results[col][metric]['means'][i]
                final_results[col][metric]['means'][i] = np.mean(scores)
                final_results[col][metric]['stds'][i] = np.std(scores)

        # Sort the lists for JSON output
        for key in ['col_list', 'lb_list', 'ub_list']:
            sorted_budgets = sorted(final_results[col][key].keys())
            final_results[col][key] = [final_results[col][key][budget] for budget in sorted_budgets]

    if type not in ['SPD', 'EO']:
        metric = 'AUC'
    else:
        metric = type
        type = 'MNAR'

    filename = f"google_results_h2o_{type}_{metric}_LR.json"
    with open(filename, 'w') as json_file:
        json.dump(final_results, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()