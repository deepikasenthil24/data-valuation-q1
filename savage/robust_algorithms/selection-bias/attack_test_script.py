import sys, os
import numpy as np
import pandas as pd
import torch
import math
import random
import itertools

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from argparse import Namespace

from models import LogisticRegression, weights_init_normal
from FairBatchSampler_Multiple import FairBatch, CustomDataset
from utils import correlation_reweighting, datasampling, test_model

import cvxopt
import cvxpy as cp
from cvxpy import OPTIMAL, Minimize, Problem, Variable, quad_form # Work in YJ kernel


from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression as SKLR
from tqdm import tqdm

sys.path.append(os.path.abspath("../../"))
os.chdir('../../')
from load_dataset import load
from API_Design_a import MissingValueError, SamplingError, LabelError, Injector


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

X_train, X_test, y_train, y_test = load('adult')
X_train_orig = X_train.copy()
X_test_orig = X_test.copy()
y_train_orig = y_train.copy()
y_test_orig = y_test.copy()

# compute correlation between label and sensitive attribute using corrcoef
corr = np.corrcoef(X_train.gender, y_train)[0, 1]
# corr = 0.18

pr_y_orig = sum(y_train == 1) / len(y_train)
pr_z_orig = sum(X_train.gender == 1) / len(X_train)

lb_list = [11, 1]
ub_list = [15, 1]
mv_pattern = create_pattern(['education', 'Y'], lb_list, ub_list)
mv_pattern_len = np.sum(mv_pattern(X_train, y_train))
poi_ratio = 0.1

mv_num = min(mv_pattern_len, int(poi_ratio*len(X_train)))
mv_err = SamplingError(mv_pattern, mv_num / mv_pattern_len)
injector = Injector(error_seq=[mv_err])
X_train, y_train, _, _ = injector.inject(X_train, y_train, X_train, y_train, seed=0)

pr_y_shifted = sum(y_train == 1) / len(y_train)
pr_z_shifted = sum(X_train.gender == 1) / len(X_train)

y_train = y_train.replace({0: -1, 1: 1})
y_test = y_test.replace({0: -1, 1: 1})

xz_train = X_train.copy()
z_train = X_train.gender.copy()
y_noise = y_train.copy()

xz_test = X_test.copy()
z_test = X_test.gender.copy()

xz_train = torch.FloatTensor(xz_train.to_numpy())
y_train = torch.FloatTensor(y_train.to_numpy())
z_train = torch.FloatTensor(z_train.to_numpy())

xz_test = torch.FloatTensor(xz_test.to_numpy())
y_test = torch.FloatTensor(y_test.to_numpy())
z_test = torch.FloatTensor(z_test.to_numpy())


print("---------- Number of Data ----------" )
print(
    "Train data : %d, Test data : %d "
    % (len(y_train), len(y_test))
)
print("------------------------------------")

w = np.array([sum((z_train==1)&(y_train==1))/len(y_train), sum((z_train==0)&(y_train==1))/len(y_train),
              sum((z_train==1)&(y_train==-1))/len(y_train), sum((z_train==0)&(y_train==-1))/len(y_train)])
alpha = 0.005 #

# test robust algo
def run_epoch(model, train_features, labels, optimizer, criterion):
    """Trains the model with the given train data.

    Args:
        model: A torch model to train.
        train_features: A torch tensor indicating the train features.
        labels: A torch tensor indicating the true labels.
        optimizer: A torch optimizer.
        criterion: A torch criterion.

    Returns:
        loss value.
    """

    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    loss = criterion((F.tanh(label_predicted.squeeze()) + 1) / 2, (labels.squeeze() + 1) / 2)
    loss.backward()

    optimizer.step()

    return loss.item()


def find_w_cvxpy(w, corr, gamma1, gamma2):
    """Solves the SDP relaxation problem.

    Args:
        w: A list indicating the original data ratio for each (y, z)-class.
        corr: A real number indicating the target correlation.
        gamma1: A real number indicating the range of Pr(y) change
        gamma2: A real number indicating the range of Pr(z) change

    Returns:
        solution for the optimization problem.
    """

    n = len(w)
    a = w[0]
    b = w[1]
    c = w[2]
    d = w[3]
    orig_corr = w[0] / (w[0] + w[2]) - w[1] / (w[1] + w[3])

    P0 = np.array([[1, 0, 0, 0, -a], [0, 1, 0, 0, -b], [0, 0, 1, 0, -c], [0, 0, 0, 1, -d], [-a, -b, -c, -d, 0]])

    P1 = np.array([[0, -corr / 2, 0, (1 - corr) / 2, 0], [-corr / 2, 0, (-1 - corr) / 2, 0, 0],
                   [0, (-1 - corr) / 2, 0, -corr / 2, 0], [(1 - corr) / 2, 0, -corr / 2, 0, 0], [0, 0, 0, 0, 0]])

    P2 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0]])
    r2 = -2 * (a + b)

    P3 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]])
    r3 = -2 * (a + c)

    P4 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 1, 1, 1, 0]])
    r4 = -2 * 1

    P5 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

    X = cp.Variable((n + 1, n + 1), symmetric=True)

    constraints = [X >> 0]
    constraints = [
        cp.trace(P1 @ X) == 0,
        cp.trace(P2 @ X) + r2 <= gamma1,
        cp.trace(P2 @ X) + r2 >= -gamma1,
        cp.trace(P3 @ X) + r3 <= gamma2,
        cp.trace(P3 @ X) + r3 >= -gamma2,
        cp.trace(P4 @ X) + r4 == 0,
        cp.trace(P5 @ X) == 1,
        X >> 0
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(P0 @ X)), constraints)

    result = prob.solve()

    x = X.value
    x = x[:, -1][:-1]
    return x


# Set the train data
train_data = CustomDataset(xz_train, y_noise, z_train)

seeds = [42, 43, 44, 45, 46]

fairshift_aucs = []
fairshift_accs = []
fairshift_f1s = []
fairshift_eos = []

reweighing_aucs = []
reweighing_accs = []
reweighing_f1s = []
reweighing_eos = []

lfr_aucs = []
lfr_accs = []
lfr_f1s = []
lfr_eos = []

train_type = 'ours'

full_tests = []
full_trains = []

""" Find new data ratio for each (y, z)-class """
gamma_y = abs(pr_y_shifted - pr_y_orig)
gamma_z = abs(pr_z_shifted - pr_z_orig)
w_new = find_w_cvxpy(w, corr, gamma_y, gamma_z)

""" Find example weights according to the new weight """
our_weights = correlation_reweighting(xz_train, y_train, z_train, w, w_new)

""" Train models """
for seed in tqdm(seeds):

    print("< Seed: {} >".format(seed))

    # ---------------------
    #  Initialize model, optimizer, and criterion
    # ---------------------

    useCuda = False
    if useCuda:
        model = LogisticRegression(xz_train.shape[1], 1).cuda()
    else:
        model = LogisticRegression(xz_train.shape[1], 1)

    torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []

    # ---------------------
    #  Set data and batch sampler
    # ---------------------

    if train_type == 'in-processing-only':
        train_data = CustomDataset(xz_train, y_train, z_train)
    else:
        new_index = datasampling(xz_train, y_train, z_train, our_weights, seed=seed)
        train_data = CustomDataset(xz_train[new_index], y_train[new_index], z_train[new_index])

    sampler = FairBatch(model, train_data.x, train_data.y, train_data.z, batch_size=100, alpha=alpha,
                        target_fairness='eqopp', replacement=False, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------

    for epoch in range(500):
        print(epoch, end="\r")

        tmp_loss = []

        for batch_idx, (data, target, z) in enumerate(train_loader):
            loss = run_epoch(model, data, target, optimizer, criterion)
            tmp_loss.append(loss)

        losses.append(sum(tmp_loss) / len(tmp_loss))

    pred_digits = model(xz_test).detach().numpy()
    idx_privileged = np.where((X_test.gender == 1).to_numpy() & (y_test == 1).detach().numpy())[0]
    tpr_privileged = np.mean((pred_digits > 0)[idx_privileged])
    idx_protected = np.where((X_test.gender == 0).to_numpy() & (y_test == 1).detach().numpy())[0]
    tpr_protected = np.mean((pred_digits > 0)[idx_protected])
    eq_opp = abs(tpr_privileged - tpr_protected)
    auc = roc_auc_score(y_test_orig, (1/(1+np.exp(-pred_digits))).ravel())

    print("----------------------------------------------------------------------")
    print('Robust Algo:')
    print(f"Test AUC: {auc}, EO: {eq_opp}")
    print("----------------------------------------------------------------------")

    fairshift_aucs.append(auc)
    fairshift_eos.append(eq_opp)

    # test reweighing

    # use the same dataset
    privileged_groups = [{'gender': 1}]
    unprivileged_groups = [{'gender': 0}]
    X_train_reweighed = BinaryLabelDataset(df=pd.concat([X_train, pd.Series(y_train.detach().numpy(), name='Y')], axis=1),
                                           label_names=['Y'], protected_attribute_names=['gender'],
                                           favorable_label=1, unfavorable_label=-1)

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    RW.fit(X_train_reweighed)
    X_train_reweighed = RW.transform(X_train_reweighed)

    clf = SKLR(random_state=seed)
    clf.fit(X_train_reweighed.features, X_train_reweighed.labels, sample_weight=X_train_reweighed.instance_weights)
    y_pred = clf.predict_proba(X_test)[:, 1]

    # measure equal opportunity
    eo = np.mean((y_pred>0.5)[X_test_orig[((X_test_orig['gender'] == 1) & (y_test_orig == 1))].index]) - \
         np.mean((y_pred>0.5)[X_test_orig[((X_test_orig['gender'] == 0) & (y_test_orig == 1))].index])
    eo = abs(eo)
    auc = roc_auc_score(y_test_orig, y_pred)

    print("----------------------------------------------------------------------")
    print('Reweighing:')
    print(f"Test AUC: {auc}, EO: {eo}")
    print("----------------------------------------------------------------------")

    reweighing_aucs.append(auc)
    reweighing_eos.append(eo)

    # test LFR

    X_train_wrapped = BinaryLabelDataset(df=pd.concat([X_train, pd.Series(y_train.detach().numpy(), name='Y')], axis=1),
                                         label_names=['Y'], protected_attribute_names=['gender'],
                                         favorable_label=1, unfavorable_label=-1)
    X_test_wrapped = BinaryLabelDataset(df=pd.concat([X_test, pd.Series(y_test.detach().numpy(), name='Y')], axis=1),
                                        label_names=['Y'], protected_attribute_names=['gender'],
                                        favorable_label=1, unfavorable_label=-1)

    preproc = LFR(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, k=10, Ax=0.1, Ay=2.0,
                  Az=1.0, verbose=1, seed=seed)
    preproc.fit(X_train_wrapped, maxiter=3000, maxfun=3000)
    X_train_transformed = preproc.transform(X_train_wrapped).convert_to_dataframe()[0]
    X_test_transformed = preproc.transform(X_test_wrapped).convert_to_dataframe()[0]

    clf = SKLR(random_state=seed)
    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict_proba(X_test_transformed)[:, 1]

    # measure equal opportunity
    eo = np.mean((y_pred>0.5)[X_test_orig[((X_test_orig['gender'] == 1) & (y_test_orig == 1))].index]) - \
         np.mean((y_pred>0.5)[X_test_orig[((X_test_orig['gender'] == 0) & (y_test_orig == 1))].index])
    eo = abs(eo)
    auc = roc_auc_score(y_test_orig, y_pred)

    print("----------------------------------------------------------------------")
    print('LFR:')
    print(f"Test AUC: {auc}, EO: {eo}")
    print("----------------------------------------------------------------------")

    lfr_aucs.append(auc)
    lfr_eos.append(eo)

# print mean and std
print("----------------------------------------------------------------------")
print('FairShift:')
print(f"Test AUC: {np.mean(fairshift_aucs)} +- {np.std(fairshift_aucs)}")
print(f"EO: {np.mean(fairshift_eos)} +- {np.std(fairshift_eos)}")
print("----------------------------------------------------------------------")

print("----------------------------------------------------------------------")
print('Reweighing:')
print(f"Test AUC: {np.mean(reweighing_aucs)} +- {np.std(reweighing_aucs)}")
print(f"EO: {np.mean(reweighing_eos)} +- {np.std(reweighing_eos)}")
print("----------------------------------------------------------------------")

print("----------------------------------------------------------------------")
print('LFR:')
print(f"Test AUC: {np.mean(lfr_aucs)} +- {np.std(lfr_aucs)}")
print(f"EO: {np.mean(lfr_eos)} +- {np.std(lfr_eos)}")
print("----------------------------------------------------------------------")