import copy
import optuna
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_dataset import load
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from API_Design_a import MissingValueError, LabelError, SamplingError, Injector
import argparse
import os
import json
import itertools

def initialize_model_and_baseline_auc(dataset, model_name, cleaning):
    X_train, X_test, y_train, y_test = load(dataset)

    X_train_orig = copy.deepcopy(X_train)
    X_test_orig = copy.deepcopy(X_test)
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(
        X_train_orig, y_train, test_size=0.25, random_state=42
    )

    X_train_orig, X_val_orig, X_test_orig = (
        X_train_orig.reset_index(drop=True),
        X_val_orig.reset_index(drop=True),
        X_test_orig.reset_index(drop=True),
    )
    y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

    if cleaning == 'h2o':
        ss = StandardScaler()
        ss.fit(X_train_orig)
        X_train_scaled = ss.transform(X_train_orig)
        X_test_scaled = ss.transform(X_test_orig)

        clf = get_model(model_name)
        clf.fit(X_train_scaled, y_train)

        y_test_pred = clf.predict_proba(X_test_scaled)[:, 1]
        baseline_auc = roc_auc_score(y_test, y_test_pred)
        print(f"Baseline AUC: {baseline_auc}")

        # 计算特征重要性
        feature_importance = compute_feature_importance(X_train_orig, y_train)
        feature_importance.to_csv('feature_importances.csv')
    else:
        raise ValueError("Unsupported cleaning approach name")

    return baseline_auc, X_train_orig, y_train

def get_model(model_name):
    if model_name == 'LR':
        return LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == 'DT':
        return DecisionTreeClassifier(random_state=42)
    elif model_name == 'NN':
        return MLPClassifier(random_state=42, hidden_layer_sizes=(10,))
    elif model_name == 'RF':
        return RandomForestClassifier(random_state=42)
    elif model_name == 'SVM':
        sgd_clf = SGDClassifier(loss='hinge', random_state=42, max_iter=1000, tol=1e-3)
        return CalibratedClassifierCV(sgd_clf, method='sigmoid', cv=5)
    else:
        raise ValueError("Unsupported model name")

def compute_feature_importance(X, y):
    # 计算特征重要性（使用随机森林）
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    return feature_importance

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
                    value = data_y.loc[i]
                else:
                    value = data_X.loc[i, col_list[j]]

                if (value < lb_list[j]) or (value > ub_list[j]):
                    satisfaction = False
                    break

            if satisfaction:
                binary_indicators.append(1)
            else:
                binary_indicators.append(0)
        return np.array(binary_indicators)

    return pattern

def objective_fuc(trial, col_id, budget, baseline_auc, pattern_col_list, error_type, dataset, model_name, cleaning, X_train_orig, y_train):
    lb_list = []
    ub_list = []
    X_test_orig, y_test = load(dataset)[1], load(dataset)[3]

    X_train_orig.reset_index(drop=True, inplace=True)
    X_test_orig.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    current_pattern_col_list = pattern_col_list

    # Generate lb_list and ub_list
    for col in current_pattern_col_list:
        col_data = y_train if col == 'Y' else X_train_orig[col]
        col_min = col_data.min()
        col_max = col_data.max()
        col_range = col_max - col_min

        lb_percent = trial.suggest_float(f'{col}_lb_percent', 0, 1)
        ub_percent = trial.suggest_float(f'{col}_ub_percent', lb_percent, 1)

        lb = col_min + lb_percent * col_range
        ub = col_min + ub_percent * col_range

        lb_list.append(lb)
        ub_list.append(ub)

    mv_pattern = create_pattern(current_pattern_col_list, lb_list, ub_list)
    mv_pattern_len = np.sum(mv_pattern(X_train_orig, y_train))

    if mv_pattern_len == 0:
        # 返回一个中性值，避免立即剪枝
        return 1

    mv_num = min(mv_pattern_len, budget)

    # Define error injection
    if error_type == 'Label':
        label_flip_ratio = mv_num / mv_pattern_len
        mv_err = LabelError(pattern=mv_pattern, ratio=label_flip_ratio)
    elif error_type == 'Sampling':
        sampling_error_ratio = mv_num / mv_pattern_len
        mv_err = SamplingError(pattern=mv_pattern, ratio=sampling_error_ratio)
    else:
        mv_err = MissingValueError(col_id, mv_pattern, mv_num / mv_pattern_len)

    injecter = Injector(error_seq=[mv_err])
    dirty_X_train_orig, dirty_y_train, _, _ = injecter.inject(X_train_orig.copy(), y_train.copy(), X_train_orig,
                                                              y_train, seed=42)

    if error_type == 'Label' and len(np.unique(dirty_y_train)) < 2:
        # 返回一个中性值，避免立即剪枝
        return 1

    # Cleaning and modeling
    try:
        dirty_X_train_imputed = SimpleImputer(strategy='mean').fit_transform(dirty_X_train_orig)
        ss = StandardScaler()
        ss.fit(dirty_X_train_imputed)
        X_train_scaled = ss.transform(dirty_X_train_imputed)
        X_test_scaled = ss.transform(X_test_orig)

        clf = get_model(model_name)
        clf.fit(X_train_scaled, dirty_y_train)

        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        auc_drop = auc - baseline_auc

        # 将 AUC Drop 作为中间结果，供剪枝使用
        trial.report(auc_drop, step=0)

        # 检查是否需要剪枝
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        trial.set_user_attr('auc', auc)
        return auc_drop

    except Exception as e:
        # 出现异常，提前剪枝
        raise optuna.exceptions.TrialPruned()

def run_one_setting(setting_dict):
    dataset = setting_dict['dataset']
    model_name = setting_dict['model_name']
    pattern_col_list = setting_dict['pattern_col_list']
    baseline_auc = setting_dict['baseline_auc']
    seed = setting_dict['seed']
    missing_col = setting_dict['missing_col']
    budget = setting_dict['budget']
    error_type = setting_dict['error_type']
    n_trials = setting_dict['n_trials']
    cleaning = setting_dict['cleaning']
    X_train_orig = setting_dict['X_train_orig']
    y_train = setting_dict['y_train']

    col_id = list(X_train_orig.columns).index(missing_col)
    print(f"Evaluating seed: {seed} for {missing_col} with {budget} errors...")

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        lambda trial: objective_fuc(trial, col_id, budget, baseline_auc, pattern_col_list, error_type, dataset,
                                    model_name, cleaning, X_train_orig, y_train),
        n_trials=n_trials,
        gc_after_trial=True
    )

    best_auc = study.best_trial.user_attrs.get('auc', baseline_auc)
    auc_drop = best_auc - baseline_auc

    return {
        'seed': seed,
        'missing_col': missing_col,
        'budget': budget,
        'auc': best_auc,
        'auc_drop': auc_drop,
        'col_list': pattern_col_list,
    }

def main():
    parser = argparse.ArgumentParser(description="Run data integrity attacks on classification models.")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--cleaning', type=str, required=True, help='Cleaning approach/framework')
    parser.add_argument('--model', type=str, required=True, choices=['LR', 'DT', 'NN', 'RF', 'SVM'],
                        help='Downstream model name')
    parser.add_argument('--error_type', type=str, required=True, choices=['MNAR', 'MAR', 'Sampling', 'Label'],
                        help='Error type')
    parser.add_argument('--error_pct', type=float, default=0.1, help='Error number percentage (default: 10%)')
    parser.add_argument('--sens_attr', type=str, required=True, help='Sensitive attribute')
    parser.add_argument('--objective', type=str, required=True, choices=['AUC', 'EO', 'SPD'], help='Attack objective')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna optimization')
    parser.add_argument('--n_processes', type=int, default=10,
                        help='Number of processes for multiprocessing (default: 10)')
    parser.add_argument('--override', action='store_true', help='Override existing results')
    parser.add_argument('--error_cols', type=str, nargs='+', required=True, help='Error injection columns')
    parser.add_argument('--pattern_cols', type=str, nargs='+', required=True, help='Pattern columns')

    args = parser.parse_args()

    if not os.path.exists('./save/'):
        os.makedirs('./save/')

    dataset = args.dataset
    cleaning = args.cleaning
    model_name = args.model
    error_type = args.error_type
    error_pct = args.error_pct
    sens_attr = args.sens_attr
    attack_objective = args.objective
    n_trials = args.n_trials
    processes = args.n_processes
    override = args.override

    baseline_auc, X_train_orig, y_train = initialize_model_and_baseline_auc(dataset, model_name, cleaning)

    # 使用解析的参数
    error_cols = args.error_cols
    pattern_cols = args.pattern_cols

    # 设置实验参数
    setting = {
        'dataset': dataset,
        'model_name': model_name,
        'error_type': error_type,
        'baseline_auc': baseline_auc,
        'seed': 42,  # 可以根据需要设置
        'missing_col': error_cols[0],
        'budget': int(error_pct * len(X_train_orig)),
        'n_trials': n_trials,
        'pattern_col_list': pattern_cols,
        'sens_attr': sens_attr,
        'cleaning': cleaning,
        'X_train_orig': X_train_orig,
        'y_train': y_train
    }

    # 运行实验
    results = []
    result = run_one_setting(setting)
    results.append(result)

    # 保存结果
    result_filename = f"{dataset}_results_{cleaning}_{error_type}_{attack_objective}_{model_name}_{error_cols[0]}_{'_'.join(pattern_cols)}.json"
    with open(result_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    # 打印结果
    print(f"Results saved to {result_filename}:")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    main()