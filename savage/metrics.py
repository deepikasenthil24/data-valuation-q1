from sklearn.metrics import f1_score, roc_auc_score
import torch
import numpy as np
import json

with open('config.json', 'r') as f:
    txt = f.read()
    dtype = json.loads(txt)['dtype']
    f.close()
if dtype == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


def computeAccuracy(y_true, y_pred):
    return np.sum((y_pred > 0.5) == y_true) / len(y_pred)


def computeF1(y_true, y_pred):
    return f1_score(y_true, (y_pred > 0.5).astype(int))

def computeAUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def computeFairness(y_pred, X_test, y_test, metric, dataset): 
    fairnessMetric = 0
    if dataset == 'german':
        protected_idx = X_test[X_test['age']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['age']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'compas':
        protected_idx = X_test[X_test['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'adult':
        protected_idx = X_test[X_test['gender']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['gender']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'hmda':
        protected_idx = X_test[X_test['race']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['race']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'sqf':
        protected_idx = X_test[X_test['gender']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['gender']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'law':
        protected_idx = X_test[X_test['racetxt']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['racetxt']==1].index
        numPrivileged = len(privileged_idx)
    elif dataset == 'random':
        protected_idx = X_test[X_test['AA']==0].index
        numProtected = len(protected_idx)
        privileged_idx = X_test[X_test['AA']==1].index
        numPrivileged = len(privileged_idx)

    p_protected = 0
    for i in range(len(protected_idx)):
        p_protected += y_pred[protected_idx[i]]
    p_protected /= len(protected_idx)

    p_privileged = 0
    for i in range(len(privileged_idx)):
        p_privileged += y_pred[privileged_idx[i]]
    p_privileged /= len(privileged_idx)
    
    # statistical parity difference
    statistical_parity = p_protected - p_privileged
    
    # equality of opportunity, or 
    # true positive rate parity
    # P(Y=1 | Y=1, G=0)- P(Y=1 | Y=1, G=1)
    true_positive_protected = 0
    actual_positive_protected = 0
    for i in range(len(protected_idx)):
        if (y_test[protected_idx[i]] == 1):
            actual_positive_protected += 1
            true_positive_protected += y_pred[protected_idx[i]]
    tpr_protected = true_positive_protected/actual_positive_protected

    true_positive_privileged = 0
    actual_positive_privileged = 0
    for i in range(len(privileged_idx)):
        if (y_test[privileged_idx[i]] == 1):
            actual_positive_privileged += 1
#             if (y_pred[privileged_idx[i]][1] > y_pred[privileged_idx[i]][0]):
            true_positive_privileged += y_pred[privileged_idx[i]]
    tpr_privileged = true_positive_privileged/actual_positive_privileged

    tpr_parity = tpr_protected - tpr_privileged
    
    # equalized odds or TPR parity + FPR parity
    # false positive rate parity
    
    # predictive parity
    p_o1_y1_s1 = 0
    p_o1_s1 = 0
    for i in range(len(protected_idx)):
        p_o1_s1 += y_pred[protected_idx[i]]
        if (y_test[protected_idx[i]] == 1):
            p_o1_y1_s1 += y_pred[protected_idx[i]]
    ppv_protected = p_o1_y1_s1/p_o1_s1
    
    p_o1_y1_s0 = 0
    p_o1_s0 = 0
    for i in range(len(privileged_idx)):
        p_o1_s0 += y_pred[privileged_idx[i]]
        if (y_test[privileged_idx[i]] == 1):
            p_o1_y1_s0 += y_pred[privileged_idx[i]]
    ppv_privileged = p_o1_y1_s0/p_o1_s0
    
    predictive_parity = ppv_protected - ppv_privileged
    
    if (metric == 0):
        fairnessMetric = statistical_parity
    elif (metric == 1):
        fairnessMetric = tpr_parity
    elif (metric == 2):
        fairnessMetric = predictive_parity
        
    return fairnessMetric

