import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from opendataval.dataval import DataEvaluator
from sklearn.linear_model import LogisticRegression as SKLR
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from typing import Optional


class Kairos(DataEvaluator):
    def __init__(self, lambda_weight=0.97, sigma_feature=3, kernel_type='sigma', unbiased=False):
        """
        Args:
            lambda_weight (float): Weight for the residual squared term.
            sigma_feature (float): Bandwidth for the Gaussian (RBF) kernel on features.
            kernel_type (str): Only 'sigma' is supported (with fixed parameters).
            
        Note:
            No training (optimization) is performed in this class.
            The evaluation metric for each training sample is defined as:
            
                feature_metric + lambda_weight * (squared residual)
                
            where:
                feature_metric = avg_{j in train} k(x_i, x_j) - avg_{j in valid} k(x_j, x_i),
                k is an RBF kernel with bandwidth sigma_feature, and
                squared residual = ||(y_train[i] - ŷ_train[i])||².
        """
        super().__init__()
        if kernel_type != 'sigma':
            raise ValueError("Kairos only supports kernel_type 'sigma' with fixed parameters.")
        self.lambda_weight = lambda_weight
        self.sigma_feature = sigma_feature

        # Data placeholders; these are set by input_data.
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.r_train = None  # residual for training samples

        # These will be computed in train_data_values
        self.avg_K_train = None
        self.avg_K_valid = None

        self.unbiased = unbiased

    def input_data(self, x_train, y_train, x_valid, y_valid) -> "Kairos":
        """
        Prepares the training and validation data and computes residuals on the training set.
        The classifier (SKLR) is trained on the validation data and then used to predict 
        on the training set (to get ŷ_train).
        """
        # Convert inputs to numpy arrays if needed.
        if isinstance(x_train, torch.Tensor):
            x_train = x_train.detach().cpu().numpy()
        if isinstance(x_valid, torch.Tensor):
            x_valid = x_valid.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()
        if isinstance(y_valid, torch.Tensor):
            y_valid = y_valid.detach().cpu().numpy()

        x_train = np.array(x_train, dtype=np.float32)
        x_valid = np.array(x_valid, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_valid = np.array(y_valid, dtype=np.float32)

        self.input_dim = x_train.shape[1]
        #self.num_classes = y_train.shape[1]
        # If labels are one-hot encoded
        if y_train.ndim == 2:
            self.num_classes = y_train.shape[1]
        # If labels are class indices
        else:
            self.num_classes = int(np.max(y_train)) + 1

        # Convert to torch tensors.
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.X_valid = torch.tensor(x_valid, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.y_valid = torch.tensor(y_valid, dtype=torch.float32)

        # Train a simple classifier (SKLR) on the validation data to obtain predicted probabilities.
        if self.lambda_weight == 1: 
            self.r_train = torch.zeros(y_train.shape, dtype=torch.float32)
        else:
            y_valid_indices = np.argmax(y_valid, axis=1)
            # self.classifier = MLPClassifier(hidden_layer_sizes=(100, 100), random_state=42)
            self.classifier = SKLR(random_state=42)
            self.classifier.fit(x_valid, y_valid_indices) 
            p_train = self.classifier.predict_proba(x_train)

            r_train = y_train - p_train 
            self.r_train = torch.tensor(r_train, dtype=torch.float32)
        return self

    def train_data_values(self, *args, **kwargs):
        """
        Pre-compute the per-sample average RBF-kernel values:
          - avg_K_train[i] = mean_j k(x_i, x_j)
          - avg_K_valid[i] = mean_j k(x_j, x_i)
    
        This version fuses the two kernel computations into one large matrix
        multiply + one exp, sharing all the common work.
        """
        X = self.X_train        # (n_train, d)
        V = self.X_valid        # (n_valid, d)
        n_train = X.shape[0]
    
        # bandwidth  
        sigma2 = self.sigma_feature ** 2
        inv_two_sigma2 = 1.0 / (2.0 * sigma2)
    
        # 1) stack the data so we do one big matmul instead of two
        Z = torch.cat([X, V], dim=0)     # shape: (n_train + n_valid, d)
    
        # 2) squared ℓ₂ norms of each row in Z
        #    so we only compute these once
        Z_norm_sq = (Z * Z).sum(dim=1)   # shape: (n_train + n_valid,)
    
        # 3) one matrix–multiply for all cross terms
        #    Z @ Xᵀ gives a (n_train+n_valid)×n_train Gram block:
        G = Z @ X.T                      # shape: (n_train+n_valid, n_train)
    
        # 4) pairwise squared distances:
        #    D[i,j] = ‖Z[i]−X[j]‖² = Z_norm_sq[i] + X_norm_sq[j] − 2·G[i,j]
        #    note: X_norm_sq is just the first n_train entries of Z_norm_sq
        D = Z_norm_sq.unsqueeze(1) + Z_norm_sq[:n_train].unsqueeze(0) - 2.0 * G
    
        # 5) kernel matrix for *all* pairs in one exp
        K = torch.exp(-D * inv_two_sigma2)   # shape: (n_train+n_valid, n_train)
    
        # 6) split back into train/valid blocks
        K_train = K[:n_train, :]             # (n_train, n_train)
        K_valid = K[n_train:, :]             # (n_valid, n_train)
    
        # 7) averages
        self.avg_K_train = K_train.mean(dim=1)    # (n_train,)
        self.avg_K_valid = K_valid.mean(dim=0)    # (n_train,)

        # feature discrepancy
        if self.unbiased:
            feature_metric = self.avg_K_train - (self.avg_K_valid * len(self.X_train) - 1) / (len(self.X_train) - 1)
        else:
            feature_metric = self.avg_K_train - self.avg_K_valid

        # squared residual term
        squared_residual = np.sqrt((self.r_train ** 2).sum(dim=1))

        self.squared_residual = squared_residual
        self.feature_metric = feature_metric
    
        return self

    def _rbf_kernel(self, A, B, sigma):
        """
        Computes the RBF kernel between each row in A and each row in B.
        """
        A_norm_sq = (A ** 2).sum(dim=1).unsqueeze(1)
        B_norm_sq = (B ** 2).sum(dim=1).unsqueeze(0)
        dist_sq = A_norm_sq + B_norm_sq - 2 * (A @ B.T)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def evaluate_data_values(self) -> np.ndarray:
        """
        Uses the pre-computed avg_K_train / avg_K_valid plus residuals
        to form the final metric per training sample.
        """
        # feature discrepancy
        if self.unbiased:
            feature_metric = self.avg_K_train - (self.avg_K_valid * len(self.X_train) - 1) / (len(self.X_train) - 1)
        else:
            feature_metric = self.avg_K_train - self.avg_K_valid

        # squared residual term
        squared_residual = np.sqrt((self.r_train ** 2).sum(dim=1))

        self.squared_residual = squared_residual
        self.feature_metric = feature_metric

        # combined metric
        metric = self.lambda_weight * feature_metric + (1 - self.lambda_weight) * squared_residual

        return -metric.detach().cpu().numpy()

    def online_update(self, x_new, y_new) -> "Kairos":
        """
        Incrementally update the values when a new batch (x_new, y_new) arrives.

        Args:
            x_new:  New training features, shape (m, d), numpy array or torch.Tensor.
            y_new:  New training labels (one-hot or probabilities), shape (m, c),
                    numpy array or torch.Tensor.

        Returns:
            self, with all internal buffers updated.
        """
        # 1) Normalize inputs to torch.float32
        import numpy as _np, torch as _torch

        # convert to numpy
        if isinstance(x_new, _torch.Tensor):
            x_new = x_new.detach().cpu().numpy()
        if isinstance(y_new, _torch.Tensor):
            y_new = y_new.detach().cpu().numpy()

        x_new = _np.array(x_new, dtype=_np.float32)
        y_new = _np.array(y_new, dtype=_np.float32)

        # to torch
        X_new = _torch.tensor(x_new, dtype=_torch.float32)
        y_new = _torch.tensor(y_new, dtype=_torch.float32)

        # 2) Compute residuals for the new batch
        if self.lambda_weight == 1:
            r_new = _torch.zeros_like(y_new)
        else:
            # classifier was trained on validation in input_data()
            p_new = self.classifier.predict_proba(x_new)  # shape (m, c)
            r_new = _torch.tensor(y_new.numpy() - p_new, dtype=_torch.float32)

        # 3) Prepare sizes
        N_old = self.X_train.shape[0]
        m = X_new.shape[0]
        N_total = N_old + m

        σ = self.sigma_feature

        # 4) Update avg_K_train for old points:
        #    avg_K_train_old_new = mean_j∈new k(x_i_old, x_j_new)
        K_old_new = self._rbf_kernel(self.X_train, X_new, σ)    # (N_old, m)
        sum_old_old = self.avg_K_train * N_old                  # (N_old,)
        sum_old_new = K_old_new.sum(dim=1)                      # (N_old,)
        avg_old_updated = (sum_old_old + sum_old_new) / N_total # (N_old,)

        # 5) Compute avg_K_train for new points:
        #    sum over old + sum over new
        K_new_old = self._rbf_kernel(X_new, self.X_train, σ)    # (m, N_old)
        K_new_new = self._rbf_kernel(X_new, X_new, σ)           # (m, m)
        sum_new = K_new_old.sum(dim=1) + K_new_new.sum(dim=1)    # (m,)
        avg_new = sum_new / N_total                             # (m,)

        # 6) Concatenate to form updated avg_K_train
        self.avg_K_train = _torch.cat([avg_old_updated, avg_new], dim=0)

        # 7) Update avg_K_valid: only need new columns
        #    avg_K_valid_new = mean_i k(x_valid_i, x_new_j)
        K_valid_new = self._rbf_kernel(self.X_valid, X_new, σ)  # (n_valid, m)
        avg_valid_new = K_valid_new.mean(dim=0)                 # (m,)
        self.avg_K_valid = _torch.cat([self.avg_K_valid, avg_valid_new], dim=0)

        # 8) Append new data to X_train, y_train, r_train
        self.X_train = _torch.cat([self.X_train, X_new], dim=0)
        self.y_train = _torch.cat([self.y_train, y_new], dim=0)
        self.r_train = _torch.cat([self.r_train, r_new], dim=0)

        return self

