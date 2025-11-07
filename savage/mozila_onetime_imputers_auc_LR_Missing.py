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
from classifier import *
from utils import *
from metrics import *  # include fairness and corresponding derivatives
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mutual_info_score, auc, roc_curve, roc_auc_score, f1_score, accuracy_score
from scipy.stats import wasserstein_distance
from optuna.samplers import *
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from API_Design_a import MissingValueError, LabelError, SamplingError, Injector

import learn2clean
import learn2clean.loading.reader as rd
import learn2clean.qlearning.qlearner as ql

import os
import json
if not(os.path.exists('./save/')):
    os.system('mkdir ./save/')

dataset = 'mozilla'
# sens_attr = 'gender'


imputers = {
    'MeanImputer': SimpleImputer(strategy='mean'),
    'MedianImputer': SimpleImputer(strategy='median'),
    'KNNImputer': KNNImputer(n_neighbors=10),
    'IterativeImputer': IterativeImputer(max_iter=10, random_state=42)
}

pattern_col_list = ['start', 'end', 'event', 'size', 'Y']
missing_col_list = ['start', 'end', 'event', 'size']
nan_counts = range(1000, 10001, 8000)
n_trials = 10
seed_values = [42, 43]


def save_results_to_json(results, filename):
    with open(filename, 'w') as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)


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


def objective(trial, col_id, budget, imputer, precedent_injection=[]):
    lb_list = []
    ub_list = []
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
    for col in pattern_col_list:
        if col == 'Y':
            mv_lb = trial.suggest_int(col+'_mv_lb', y_train.min(), y_train.max())
            lb_list.append(mv_lb)
            mv_interval = trial.suggest_int(col+'_mv_int', 0, y_train.max() - mv_lb)
            mv_ub = mv_interval + mv_lb
            ub_list.append(mv_ub)
        else:
            mv_lb = trial.suggest_int(col+'_mv_lb', X_train_orig[col].min(), X_train_orig[col].max())
            lb_list.append(mv_lb)
            mv_interval = trial.suggest_int(col+'_mv_int', 0, X_train_orig[col].max() - mv_lb)
            mv_ub = mv_interval + mv_lb
            ub_list.append(mv_ub)

    mv_pattern = create_pattern(pattern_col_list, lb_list, ub_list)
    mv_pattern_len = np.sum(mv_pattern(X_train_orig, y_train))
    if mv_pattern_len == 0:
        raise optuna.exceptions.TrialPruned()
    mv_num = min(mv_pattern_len, budget)

    mv_err = MissingValueError(col_id, mv_pattern, mv_num / mv_pattern_len)
    injecter = Injector(error_seq=precedent_injection + [mv_err])
    dirty_X_train_orig, dirty_y_train, _, _ = injecter.inject(X_train_orig.copy(), y_train.copy(),
                                                              X_train_orig, y_train, seed=42)

    X_train_imputed = pd.DataFrame(imputer.fit_transform(dirty_X_train_orig),
                                   columns=dirty_X_train_orig.columns)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_imputed, dirty_y_train)
    y_pred_proba = clf.predict_proba(X_test_orig)[:, 1]

    auc_score = roc_auc_score(y_test, y_pred_proba)

    # sensitive_group0 = (X_test_orig.gender==0)
    # sensitive_group1 = (X_test_orig.gender==1)
    # spd = np.mean(clf.predict(X_test_orig)[sensitive_group1]) -\
    #       np.mean(clf.predict(X_test_orig)[sensitive_group0])
    #
    # sensitive_group0 = (X_test_orig.gender==0)&(y_test==1)
    # sensitive_group1 = (X_test_orig.gender==1)&(y_test==1)
    # eo = np.mean(clf.predict(X_test_orig)[sensitive_group1]) -\
    #      np.mean(clf.predict(X_test_orig)[sensitive_group0])

    trial.set_user_attr('auc', auc_score)
    trial.set_user_attr('error_injector', injecter)
    # trial.set_user_attr('spd', abs(spd))
    # trial.set_user_attr('eo', abs(eo))

    return auc_score


def run_one_setting(setting_dict):
    global final_results
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
    imputer_name = setting_dict['imputer_name']
#     baseline_auc = setting_dict['baseline_auc']
    seed = setting_dict['seed']
    missing_col = setting_dict['missing_col']
    budget = setting_dict['budget']

    col_id = list(X_train_orig.columns).index(missing_col)
    print(f"Evaluating seed: {seed} for {missing_col} with {budget} NaNs...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=seed))
    study.optimize(lambda trial: objective(trial, col_id, budget, imputers[imputer_name]), n_trials=n_trials)
#     learned_err_injection_list = study.best_trial.user_attrs['error_injector'].error_seq
#     num_errs_used = study.best_trial.user_attrs['num_errs']
#     print(f"Injected {num_errs_used} errors in the this step.")
    auc = study.best_trial.user_attrs['auc']
    # spd = study.best_trial.user_attrs['spd']
    # eo = study.best_trial.user_attrs['eo']

#     if seed not in final_results[imputer_name]:
#         final_results[seed] = {}
#     if missing_col not in final_results[imputer_name][seed]:
#         final_results[seed][missing_col] = {'AUC': [], 'SPD': []}

#     final_results[imputer_name][seed][missing_col]['AUC'].append(auc)
#     final_results[imputer_name][seed][missing_col]['SPD'].append(spd)

    return seed, imputer_name, missing_col, budget, auc


if __name__ == '__main__':
    settings_list = [{'seed': seed, 'missing_col': missing_col, 'budget': budget, 'imputer_name': imputer_name}
                     for seed in seed_values for missing_col in missing_col_list
                     for budget in nan_counts for imputer_name in imputers]

    # Initialize the multiprocessing Pool
    with Pool(processes=10) as pool:
        results = pool.map(run_one_setting, settings_list)

    for imputer_name in imputers:
        final_results = dict()
        for missing_col in missing_col_list:
            final_results[missing_col] = {'AUC': dict()}
            # final_results[missing_col] = {'AUC': dict(), 'SPD': dict(), 'EO': dict()}
            for budget in nan_counts:
                final_results[missing_col]['AUC'][budget] = []
                # final_results[missing_col]['SPD'][budget] = []
                # final_results[missing_col]['EO'][budget] = []

        for seed, res_imputer_name, missing_col, budget, auc in results:
            if res_imputer_name == imputer_name:
                final_results[missing_col]['AUC'][budget].append(auc)
                # final_results[missing_col]['SPD'][budget].append(spd)
                # final_results[missing_col]['EO'][budget].append(eo)

        for missing_col in missing_col_list:
            for metric in ['AUC']:
            # for metric in ['AUC', 'SPD', 'EO']:
                means = []
                stds = []
                for budget in nan_counts:
                    means.append(np.mean(final_results[missing_col][metric][budget]))
                    stds.append(np.std(final_results[missing_col][metric][budget]))
                    del final_results[missing_col][metric][budget]

                final_results[missing_col][metric] = {'means': means, 'stds': stds, 'budgets': list(nan_counts)}

        save_results_to_json(final_results, f"final_results_{imputer_name}_MNAR_AUC_LR_mozila.json")