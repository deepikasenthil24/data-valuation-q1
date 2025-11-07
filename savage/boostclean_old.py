# Source of the following code snippet:
# Adapted from https://github.com/mohamedyd/rein-benchmark/blob/69262595d0a8ff27d165eaaad800d08967bb541d/cleaners/CPClean/code/cleaner/boost_clean.py

import copy
import numpy as np
from sklearn.metrics import mutual_info_score, auc, roc_curve, roc_auc_score, f1_score, accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(x):
    return np.log(x / 1 - x)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def train_classifiers(X_train_list, y_train, model):
    C_list = []
    for X_train in X_train_list:
        new_model = copy.deepcopy(model)
        new_model.fit(X_train, y_train)
        C_list.append(new_model)
    return C_list


def transform_y(y, c):
    y_c = copy.deepcopy(y)
    mask = y == c
    y_c[mask] = 1
    y_c[mask == False] = -1 
    return y_c


def boost_clean(model, X_train_list, y_train, X_val, y_val, X_test, y_test, X_test_sensitive, T=1):
    y_train = transform_y(y_train, 1)
    y_val = transform_y(y_val, 1)
    y_test = transform_y(y_test, 1)

    C_list = train_classifiers(X_train_list, y_train, model)
    N = len(y_val)
    W = np.ones((1, N)) / N

    preds_val = np.array([C.predict(X_val) for C in C_list]).T
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    acc_list = (preds_val == y_val).astype(int)
    C_T = []
    a_T = []
    for t in range(T):
        acc_t = W.dot(acc_list)
        c_t = np.argmax(acc_t)

        e_c = 1 - acc_t[0, c_t]
        a_t = np.log((1 - e_c) / (e_c + 1e-8))
        
        C_T.append(c_t)
        a_T.append(a_t)
        
        for i in range(N):
            W[0, i] = W[0, i] * np.exp(-a_t * y_val[i, 0] * preds_val[i, c_t])
        
        # rescale to have sum(W)=1
        W = W / np.sum(W)

    a_T = np.array(a_T).reshape(1, -1)
    a_T_proba = softmax(a_T)

    preds_test = [C.predict(X_test) for C in C_list]
    preds_test_T = np.array([preds_test[c_t] for c_t in C_T])
    test_scores = a_T.dot(preds_test_T).T

    preds_val = [C.predict(X_val) for C in C_list]
    preds_val_T = np.array([preds_val[c_t] for c_t in C_T])
    val_scores = a_T.dot(preds_val_T).T

    y_pred_test = np.sign(test_scores)
    y_pred_val = np.sign(val_scores)

    test_acc = accuracy_score(y_test, y_pred_test)
    val_acc = accuracy_score(y_val, y_pred_val)
    test_f1 = f1_score(y_test, y_pred_test)
    val_f1 = f1_score(y_val, y_pred_val)
    
    preds_proba_test = [C.predict_proba(X_test)[:, 1] for C in C_list]
    preds_proba_test_T = np.array([preds_proba_test[c_t] for c_t in C_T])
    test_proba = a_T_proba.dot(preds_proba_test_T).T

    def calculate_spd_eo(y_true, y_pred, sensitive_features):

        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
        sensitive_features = sensitive_features.ravel()

        sensitive_group0 = sensitive_features == 0
        sensitive_group1 = sensitive_features == 1

        spd = np.mean(y_pred[sensitive_group1]) - np.mean(y_pred[sensitive_group0])

        eo_group0 = sensitive_group0 & (y_true == 1)
        eo_group1 = sensitive_group1 & (y_true == 1)
        eo = np.mean(y_pred[eo_group1]) - np.mean(y_pred[eo_group0])

        return abs(spd/2), abs(eo/2)

    spd, eo = calculate_spd_eo(y_test, y_pred_test, X_test_sensitive)

    return test_acc, roc_auc_score(y_test, test_proba), spd, eo, test_f1