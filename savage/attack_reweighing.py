import copy
import optuna
import pickle
import time
import torch
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_dataset import load
from API_Design_a import MissingValueError, SamplingError, LabelError, Injector
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import mutual_info_score, auc, roc_curve, roc_auc_score, f1_score, accuracy_score
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from boostclean import boost_clean
from optuna.samplers import *

from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

dataset = 'adult'
sens_attr = 'gender'

X_train, X_test, y_train, y_test = load(dataset)

X_train_orig = copy.deepcopy(X_train)
X_test_orig = copy.deepcopy(X_test)

# Use 1/4 of training data as validation set
X_train_orig, X_val_orig, y_train, y_val = \
    train_test_split(X_train_orig, y_train, test_size=0.25, random_state=42)

X_train_orig, X_val_orig, X_test_orig = (X_train_orig.reset_index(drop=True),
                                         X_val_orig.reset_index(drop=True),
                                         X_test_orig.reset_index(drop=True))
y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)


# create pattern function given subpopulation
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


def get_model(model_name, seed=42):
    if model_name == 'LR':
        return LogisticRegression(random_state=seed)
    elif model_name == 'DT':
        return DecisionTreeClassifier(random_state=seed)
    elif model_name == 'NN':
        return MLPClassifier(random_state=seed, hidden_layer_sizes=(10,))
    elif model_name == 'RF':
        return RandomForestClassifier(random_state=seed)
    elif model_name == 'SVM':
        sgd_clf = SGDClassifier(loss='hinge', random_state=seed, max_iter=100, tol=1e-3)
        return CalibratedClassifierCV(sgd_clf, method='sigmoid', cv=5)
    else:
        raise ValueError("Unsupported model name")

clf_name = 'LR'

ss = StandardScaler()
ss.fit(X_train_orig)
X_train = ss.transform(X_train_orig)
X_test = ss.transform(X_test_orig)

clf = get_model(clf_name)
clf.fit(X_train, y_train)

# predict using the model and the testing dataset
y_test_pred = clf.predict_proba(X_test)[:, 1]

budget = len(X_train_orig)//5
# col_list = ['education', 'marital', 'gender', 'Y']
col_list = ['education', 'workclass', 'Y']

def objective_func(trial, col_id, budget=budget,
                   baseline_auc=0, pattern_col_list=['Y'], error_type='MNAR',
                   dataset='adult', model_name='LR', objective='AUC',
                   sens_attr='gender', cleaning='MeanInmputer', seed=42):
    lb_list = []
    ub_list = []
    X_train, X_test, y_train, y_test = load(dataset)
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()

    X_train_orig.reset_index(drop=True, inplace=True)
    X_test_orig.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

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
                                                              y_train, seed=seed)

    if len(np.unique(dirty_y_train)) < 2:
        raise optuna.exceptions.TrialPruned()
    trial.set_user_attr('num_errs', min(mv_pattern_len, budget))

    #     print(time.time()-t0)
    #     t0 = time.time()

    if cleaning == 'reweighing':
        privileged_groups = [{sens_attr: 1}]
        unprivileged_groups = [{sens_attr: 0}]
        X_train_reweighed = BinaryLabelDataset(
            df=pd.concat([dirty_X_train_orig, pd.Series(dirty_y_train, name='Y')], axis=1),
            label_names=['Y'], protected_attribute_names=[sens_attr],
            favorable_label=1, unfavorable_label=0)

        RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        RW.fit(X_train_reweighed)
        X_train_reweighed = RW.transform(X_train_reweighed)

        clf = get_model(model_name, seed)
        clf.fit(X_train_reweighed.features, X_train_reweighed.labels, sample_weight=X_train_reweighed.instance_weights)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # measure equal opportunity
        eo = np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
             np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
        spd = np.mean(y_pred[X_test[(X_test[sens_attr] == 1)].index]) - \
              np.mean(y_pred[X_test[(X_test[sens_attr] == 0)].index])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

    elif cleaning == 'boostclean':

        model = get_model(model_name, seed)
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

            clf = get_model(model_name, seed)
            clf.fit(X_train, dirty_y_train)

            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            if y_pred_proba.shape[1] == 1 and error_type == 'Sampling':
                raise optuna.exceptions.TrialPruned()
            else:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            spd = np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
                  np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
            eo = np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
                 np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
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
            'IterativeImputer': IterativeImputer(max_iter=10, random_state=seed)
        }
        imputer = imputers[cleaning]
        dirty_X_train_imputed = pd.DataFrame(imputer.fit_transform(dirty_X_train_orig),
                                             columns=dirty_X_train_orig.columns)

        clf = get_model(model_name, seed)
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
            spd = np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
                  np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
            eo = np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
                 np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])
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
            train_test_split(dirty_X_train_orig, dirty_y_train, test_size=0.25, random_state=seed)
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
        spd = np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 1)].index]) - \
              np.mean(y_pred[X_test_orig[(X_test_orig[sens_attr] == 0)].index])
        eo = np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 1) & (y_test == 1))].index]) - \
             np.mean(y_pred[X_test_orig[((X_test_orig[sens_attr] == 0) & (y_test == 1))].index])

    elif cleaning == 'diffprep':
        from diffprep.utils import SummaryWriter
        from diffprep.prep_space import space
        from diffprep.experiment.diffprep_experiment import DiffPrepExperiment
        from diffprep.pipeline.diffprep_fix_pipeline import DiffPrepFixPipeline
        from diffprep.trainer.diffprep_trainer import DiffPrepSGD
        from diffprep.model import LogisticRegression as DiffprepLogisticRegression
        from diffprep.experiment.experiment_utils import min_max_normalize, set_random_seed

        X_train, X_val, y_train, y_val = train_test_split(
            dirty_X_train_orig, dirty_y_train, test_size=0.5, random_state=seed
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

def get_combinations(x, comb_nums=[1, 2]):
    combinations = []
    for r in comb_nums:
        combinations.extend([list(c) for c in itertools.combinations(x, r)])
    return combinations

# beam search (if must include Y)
all_columns = list(X_train_orig.columns)

for pct in tqdm([0.1]):
    top_candidates = [(list(), 1)]
    budget = int(len(X_train)*pct)
    all_columns_res = dict()
    round_num = 1
    while round_num < 4:
        # grow top candidates
        candidates_this_round = []
        for candidate in top_candidates:
            candidate_col_list = candidate[0]
            for col in all_columns:
                if col in candidate_col_list:
                    continue
                new_col_list = candidate_col_list + [col]
                sorted_new_col_list = tuple(sorted(list(set(new_col_list))))
                if sorted_new_col_list in all_columns_res:
                    continue
                else:
                    # assume covariate shift,
                    # if not, use list(sorted_new_col_list) + ['Y'] for pattern_col_list
                    print(list(sorted_new_col_list))
                    # optimize for injection (4rd column w/ id=3: marital)
                    study = optuna.create_study(sampler=TPESampler(seed=42))
                    study.optimize(lambda trial: objective_func(trial, budget=budget, col_id=0, objective='EO',
                                                                cleaning='reweighing', error_type='Label',
                                                                model_name=clf_name,
                                                                pattern_col_list=list(sorted_new_col_list) + ['Y']),
                                   n_trials=30)
                    try:
                        num_errs_total = study.best_trial.user_attrs['num_errs']
                    except:
                        continue
                    print(f"Injected {num_errs_total} errors in total.")
                    all_columns_res[sorted_new_col_list] = study.best_trial.value
                candidates_this_round.append((new_col_list, study.best_trial.value))
        top_candidates = sorted(candidates_this_round, key=lambda x: x[1])[:3]
        print('----------- ROUND BEST -----------')
        print(top_candidates)
        print('----------------------------------')
        round_num += 1
    print(f'PCT: {pct}')
    print('TOP-5 PATTERN:')
    print([(x[0] + ('Y',), x[1]) for x in sorted(list(all_columns_res.items()), key=lambda x: x[1])][:5])