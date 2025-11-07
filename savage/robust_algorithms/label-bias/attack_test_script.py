import sys, os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch

from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression as SKLR
from argparse import Namespace
from tqdm import tqdm

from FairRobustSampler import FairRobust, CustomDataset
from models import LogisticRegression, weights_init_normal, test_model

sys.path.append(os.path.abspath("../../"))
os.chdir('../../')
from load_dataset import load
from metrics import computeF1, computeAccuracy
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

lb_list = [14, 1]
ub_list = [15, 1]
mv_pattern = create_pattern(['education', 'Y'], lb_list, ub_list)
mv_pattern_len = np.sum(mv_pattern(X_train, y_train))
poi_ratio = 0.1
mv_num = min(mv_pattern_len, int(poi_ratio*len(X_train)))
mv_err = LabelError(mv_pattern, mv_num / mv_pattern_len)
injector = Injector(error_seq=[mv_err])
X_train, y_train, _, _ = injector.inject(X_train, y_train, X_train, y_train, seed=0)

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

y_noise = torch.FloatTensor(y_noise.to_numpy())

xz_test = torch.FloatTensor(xz_test.to_numpy())
y_test = torch.FloatTensor(y_test.to_numpy())
z_test = torch.FloatTensor(z_test.to_numpy())

print("---------- Number of Data ----------" )
print(
    "Train data : %d, Test data : %d "
    % (len(y_train), len(y_test))
)
print("------------------------------------")


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
        loss values.
    """

    optimizer.zero_grad()

    label_predicted = model.forward(train_features)
    loss = criterion((F.tanh(label_predicted.squeeze()) + 1) / 2, (labels.squeeze() + 1) / 2)
    loss.backward()

    optimizer.step()

    return loss.item()

full_tests = []

parameters = Namespace(warm_start=100, tau=1-poi_ratio, alpha=0.001, batch_size=100)

# Set the train data
train_data = CustomDataset(xz_train, y_noise, z_train)

seeds = [42, 43, 44, 45, 46]

sampler_aucs = []
sampler_eos = []

reweighing_aucs = []
reweighing_eos = []

lfr_aucs = []
lfr_eos = []

for seed in tqdm(seeds):
    # ---------------------
    #  Initialize model, optimizer, and criterion
    # ---------------------

    model = LogisticRegression(xz_train.shape[1], 1)

    torch.manual_seed(seed)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()

    losses = []

    # ---------------------
    #  Define FairRobust and DataLoader
    # ---------------------

    sampler = FairRobust(model, train_data.x, train_data.y, train_data.z, target_fairness='eqopp', parameters=parameters,
                         replacement=False, seed=seed)
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, num_workers=0)

    # ---------------------
    #  Model training
    # ---------------------
    for epoch in range(400):
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

    sampler_aucs.append(auc)
    sampler_eos.append(eq_opp)

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
    clf.fit(X_train, y_train, sample_weight=X_train_reweighed.instance_weights)
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
print('FairSampler:')
print(f"Test AUC: {np.mean(sampler_aucs)} +- {np.std(sampler_aucs)}")
print(f"EO: {np.mean(sampler_eos)} +- {np.std(sampler_eos)}")
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

