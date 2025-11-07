import copy

import numpy as np
from sklearn.metrics import roc_auc_score
from autosklearn.classification import AutoSklearnClassifier
from tqdm import tqdm
from autosklearn_add_custom_clfs import add_clf


def RandomSearch(X_train, y_train, X_val, y_val, overall_time=1080, per_run_time=60, n_trials=10,
                 clf_name='LogisticRegression', mem_limit=1024*8, seed=42):
    if not(clf_name in ['LogisticRegression', 'DecisionTree', 'RandomForest', 'MLPClassifier', 'SVM']):
        raise NotImplementedError

    if clf_name == 'SVM':
        clf = AutoSklearnClassifier(time_left_for_this_task=overall_time, per_run_time_limit=per_run_time,
                                    ensemble_size=1, include={'classifier': ['liblinear_svc']}, memory_limit=mem_limit,
                                    initial_configurations_via_metalearning=0, delete_tmp_folder_after_terminate=False)
    else:
        from autosklearn_add_custom_clfs import add_clf
        clf_name = 'Custom' + clf_name
        add_clf(clf_name)
        clf = AutoSklearnClassifier(time_left_for_this_task=overall_time, per_run_time_limit=per_run_time,
                                    ensemble_size=1, include={'classifier': [clf_name]},
                                    delete_tmp_folder_after_terminate=False,
                                    initial_configurations_via_metalearning=0)
    clf_config_space = clf.get_configuration_space(X_train, y_train)
    clf_config_space.seed(seed)
    sample_config_ls = clf_config_space.sample_configuration(n_trials)
    val_aucs = []

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError(
            "Training or validation data has only one class present, cannot proceed with ROC AUC computation.")

    for i in tqdm(range(n_trials)):
        sample_config = sample_config_ls[i]
        fitted_clf, run_info, run_value = clf.fit_pipeline(X_train, y_train, config=sample_config,
                                                           X_test=X_val, y_test=y_val)
        print(fitted_clf, run_info, run_value)
        if fitted_clf:
            y_val_pred = fitted_clf.predict(X_val)
            y_val_pred_proba = fitted_clf.predict_proba(X_val)
            if y_val_pred_proba.shape[1] == 1:
                continue
            val_aucs.append((copy.deepcopy(fitted_clf), roc_auc_score(y_val, y_val_pred_proba[:, 1])))
    print(val_aucs)
    return sorted(val_aucs, key=lambda x: x[1])[-1]
