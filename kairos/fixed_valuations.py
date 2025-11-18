from typing import Optional
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from numpy.random import RandomState
from sklearn.utils import check_random_state

# Use the original OTDD library directly.
from otdd.pytorch.distance_fast import DatasetDistance, FeatureCost

from opendataval.dataval.api import DataEvaluator, ModelLessMixin
from opendataval.model import Model

from utils import torch_subset_to_tensor


def macos_fix():
    """Fix for a known Geomloss bug on MacOS.

    See: https://github.com/NVlabs/stylegan3/issues/75
    """
    import os
    import sys
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class FixedLavaEvaluator(DataEvaluator, ModelLessMixin):
    """Data valuation using the OTDD exact method.

    This evaluator computes class-wise Wasserstein distances between training and
    validation embeddings using the exact OTDD method. The computed dual solution is
    then calibrated to yield data values.

    Parameters
    ----------
    device : torch.device, optional
        Tensor device for acceleration, by default torch.device("cpu")
    random_state : RandomState, optional
        Random initial state, by default None

    Notes
    -----
    This evaluator assumes that the inputs (self.x_train, self.x_valid) are already
    computed embeddings. It also assumes that self.y_train and self.y_valid are one-hot
    encoded; however, for the OTDD routines, we convert them on the fly to integer indices.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        random_state: Optional[RandomState] = None,
        lam_x = 1,
        lam_y = 1,
        entreg=1e-1
    ):
        macos_fix()
        torch.manual_seed(check_random_state(random_state).tomaxint())
        self.device = device
        self.λ_x = lam_x
        self.λ_y = lam_y
        self.device = device
        self.entreg = entreg
        # The following attributes must be set externally before calling train_data_values:
        #   self.x_train, self.y_train, self.x_valid, self.y_valid

    def train_data_values(self, *args, **kwargs):
        """Trains the evaluator to compute data values using the OTDD exact method.

        It constructs a DatasetDistance instance from the original OTDD library and
        computes the dual solution.
        """
        # feature_cost = "euclidean"

        # Assume self.x_train and self.x_valid are already embeddings.
        x_train, x_valid = self.x_train, self.x_valid
        x_dim = len(x_train[0])
        feature_cost = FeatureCost(
            src_embedding=None,
            src_dim=(x_dim,),
            tgt_embedding=None,
            tgt_dim=(x_dim,),
            p=2,
            device=self.device,
        )

        # Convert one-hot encoded labels to integer indices on the fly,
        # without modifying self.y_train and self.y_valid.
        if self.y_train.ndim > 1:
            y_train_idx = self.y_train.argmax(dim=1)
        else:
            y_train_idx = self.y_train

        if self.y_valid.ndim > 1:
            y_valid_idx = self.y_valid.argmax(dim=1)
        else:
            y_valid_idx = self.y_valid

        # Define classes
        classes = torch.arange(0, len(self.y_train[0]))

        x_train = torch.stack([t for t in x_train])
        x_valid = torch.stack([t for t in x_valid])

        # Construct the DatasetDistance object using the exact OTDD settings.
        dist = DatasetDistance(
            (x_train, y_train_idx, classes),
            (x_valid, y_valid_idx, classes),
            inner_ot_method="exact",   # use the exact solver
            debiased_loss=True,
            feature_cost=feature_cost,
            λ_x=self.λ_x,
            λ_y=self.λ_y,
            sqrt_method="spectral",    # use spectral method for matrix square root
            sqrt_niters=10,
            precision="single",
            p=2,
            entreg=self.entreg,
            device=self.device,
            min_labelcount=2,
        )
        # Compute the dual solution.
        self.dual_sol = dist.dual_sol(maxsamples=100000, return_coupling=True)
        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return calibrated data values for each training data point.

        The calibrated gradient is computed from the dual solution, with a sign flip
        so that lower values indicate more detrimental data points.
        """
        f1k = self.dual_sol[0].squeeze()
        num_points = len(f1k) - 1
        # Calibrate the dual solution (scale and center).
        train_gradient = f1k * (1 + 1 / num_points) - f1k.sum() / num_points
        # print(sorted(train_gradient))
        # Multiply by -1 to align with data valuation conventions.
        train_gradient = -1 * train_gradient
        return train_gradient.cpu().numpy().flatten()


class FixedKNNShapley(DataEvaluator, ModelLessMixin):
    """Data valuation using KNNShapley implementation.

    KNN Shapley is a model-less mixin. This means we cannot specify an underlying
    prediction model for the DataEvaluator. However, we can specify a pretrained
    embedding model.

    References
    ----------
    .. [1] R. Jia et al.,
        Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1908.08619.

    Parameters
    ----------
    k_neighbors : int, optional
        Number of neighbors to group the data points, by default 10
    batch_size : int, optional
        Batch size of tensors to load at a time during training, by default 32
    embedding_model : Model, optional
        Pre-trained embedding model used by DataEvaluator, by default None
    random_state : RandomState, optional
        Random initial state, by default None
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        batch_size: int = 32,
        embedding_model: Optional[Model] = None,
        random_state: Optional[RandomState] = None,
    ):
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.random_state = check_random_state(random_state)

    def match(self, y: torch.Tensor) -> torch.Tensor:
        """:math:`1.` for all matching rows and :math:`0.` otherwise."""
        return (y == self.y_valid).all(dim=1).float()

    def train_data_values(self, *args, **kwargs):
        """Trains model to predict data values.

        Computes KNN shapley data values, as implemented by the following. Ignores all
        positional and key word arguments.

        References
        ----------
        .. [1] PyTorch implementation
            <https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/KNN_Shapley.py>
        """
        n = len(self.x_train)
        m = len(self.x_valid)
        x_train, x_valid = self.embeddings(self.x_train, self.x_valid)

        x_train, x_valid = torch_subset_to_tensor(x_train), torch_subset_to_tensor(x_valid)

        # Computes Euclidean distance by computing crosswise per batch
        # Doesn't shuffle to maintain relative order
        x_train_view, x_valid_view = x_train.view(n, -1), x_valid.view(m, -1)

        dist_list = []  # Uses batching to only load at most `batch_size` tensors
        for x_train_batch in DataLoader(x_train_view, self.batch_size):  # No shuffle
            dist_row = []
            for x_val_batch in DataLoader(x_valid_view, self.batch_size):
                dist_row.append(torch.cdist(x_train_batch, x_val_batch))
            dist_list.append(torch.cat(dist_row, dim=1))
        dist = torch.cat(dist_list, dim=0)

        # Arranges by distances
        sort_indices = torch.argsort(dist, dim=0, stable=True)
        y_train_sort = self.y_train[sort_indices]

        score = torch.zeros_like(dist)
        score[sort_indices[n - 1], range(m)] = self.match(y_train_sort[n - 1]) / n

        # fmt: off
        for i in tqdm(range(n - 2, -1, -1)):
            score[sort_indices[i], range(m)] = (
                score[sort_indices[i + 1], range(m)]
                + min(self.k_neighbors, i + 1) / (self.k_neighbors * (i + 1))
                * (self.match(y_train_sort[i]) - self.match(y_train_sort[i + 1]))
            )

        self.data_values = score.mean(axis=1).detach().numpy()

        return self

    def evaluate_data_values(self) -> np.ndarray:
        """Return data values for each training data point.

        Compute data values using KNN Shapley data valuation

        Returns
        -------
        np.ndarray
            Predicted data values/selection for training input data point
        """
        return self.data_values