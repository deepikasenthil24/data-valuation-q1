import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from opendataval.dataval import DataEvaluator
from opendataval.dataloader import DataFetcher
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Dict, List, Optional


def get_discover_corrupted_sample_results(
    evaluator: DataEvaluator,
    fetcher = None,
    data = None,
    percentile = 0.05,
) -> dict[str, list[float]]:
    """Evaluate discovery of noisy indices in low data value points.

    Repeatedly explores ``percentile`` of the data values and determines
    if within that total percentile, what proportion of the noisy indices are found.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested
    fetcher : DataFetcher, optional
        DataFetcher containing noisy indices, by default None
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training and
        test data with the following keys:

        - **"x_train"** Training covariates
    percentile : float, optional
        Percentile of data points to additionally search per iteration, by default .05
        
    Returns
    -------
    Dict[str, list[float]]
        dict containing list of the proportion of noisy indices found after exploring
        the ``(i * percentile)`` least valuable data points. If plot is not None,
        also returns optimal and random search performances as lists

        - **"axis"** -- Proportion of data values explored currently.
        - **"corrupt_found"** -- Proportion of corrupted data values found currently
        - **"optimal"** -- Optimal proportion of corrupted values found currently
            meaning if the inspected **only** contained corrupted samples until
            the number of corrupted samples are completely exhausted.
        - **"random"** -- Random proportion of corrupted samples found, meaning
            if the data points were explored randomly, we'd expect to find
            corrupted_samples in proportion to the number of corruption in the data set.
    """
    if isinstance(fetcher, DataFetcher):
        x_train, *_ = fetcher.datapoints
    else:
        x_train = data["x_train"]
    noisy_train_indices = fetcher.noisy_train_indices
    data_values = evaluator.data_values

    num_points = len(x_train)
    num_period = max(round(num_points * percentile), 5)  # Add at least 5 per bin
    num_bins = int(num_points // num_period) + 1

    sorted_value_list = np.argsort(data_values, kind="stable")  # Order descending
    noise_rate = len(noisy_train_indices) / len(data_values)

    # Output initialization
    found_rates = []

    # For each bin
    for bin_index in range(0, num_points + num_period, num_period):
        # from low to high data values
        found_rates.append(
            len(np.intersect1d(sorted_value_list[:bin_index], noisy_train_indices))
            / len(noisy_train_indices)
        )

    x_axis = [i / num_bins for i in range(len(found_rates))]
    eval_results = {"corrupt_found": found_rates, "axis": x_axis}

    # Returns True Positive Rate of corrupted label discovery
    return eval_results


def get_discover_corrupted_sample_results_pct(
    evaluator: 'DataEvaluator',
    fetcher: Optional['DataFetcher'] = None,
    data: Optional[Dict[str, Any]] = None,
    percentiles_to_evaluate: List[float] = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
) -> Dict[str, List[float]]:
    """Evaluate discovery of noisy indices in low data value points at specific percentiles.

    Evaluates at each percentile specified in ``percentiles_to_evaluate`` what proportion
    of the total noisy indices are found within that lowest percentile of data values.

    Parameters
    ----------
    evaluator : DataEvaluator
        DataEvaluator to be tested, containing evaluated `data_values`.
    fetcher : DataFetcher, optional
        DataFetcher containing noisy indices (`noisy_train_indices`) and training data.
        Required if `data` is not provided. By default None.
    data : dict[str, Any], optional
        Alternatively, pass in dictionary instead of a DataFetcher with the training
        data. Required if `Workspaceer` is not provided. Must contain:
        - **"x_train"**: Training covariates
        Note: `noisy_train_indices` must still be accessible via `Workspaceer` even if
              `data` is provided for `x_train`. Consider refactoring if this
              isn't the intended design. (Assuming fetcher always provides noisy indices).
    percentiles_to_evaluate : list[float], optional
        List of percentiles of data points (ranked lowest to highest value)
        at which to evaluate the discovery rate of noisy samples.
        Defaults to [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0].

    Returns
    -------
    Dict[str, list[float]]
        dict containing list of the proportion of noisy indices found after exploring
        the lowest `p` percent of data points, for each `p` in `percentiles_to_evaluate`.
        If plot is not None, also returns optimal and random search performances as lists.

        - **"axis"** -- List of percentiles evaluated (corresponds to `percentiles_to_evaluate`, sorted).
        - **"corrupt_found"** -- Proportion of corrupted data values found at each percentile in "axis".
        # Note: The 'optimal' and 'random' calculations from the original docstring
        # are not implemented in the provided *original* code. If needed, they
        # would require separate implementation here as well. The description
        # below assumes they might be added later, consistent with the original docstring style.
        # - **"optimal"** -- Optimal proportion of corrupted values found currently
        #        meaning if the inspected **only** contained corrupted samples until
        #        the number of corrupted samples are completely exhausted.
        # - **"random"** -- Random proportion of corrupted samples found, meaning
        #        if the data points were explored randomly, we'd expect to find
        #        corrupted_samples in proportion to the number of corruption in the data set.
    """
    if fetcher is None and data is None:
         raise ValueError("Either 'fetcher' or 'data' must be provided.")
    if evaluator is None:
        raise ValueError("'evaluator' must be provided.")
        
    # Get training data and noisy indices
    if isinstance(fetcher, DataFetcher): # Assuming DataFetcher class exists
        x_train = fetcher.datapoints[0] # Assuming datapoints returns (x_train, ...)
        # Assuming noisy_train_indices are directly accessible from fetcher
        if not hasattr(fetcher, 'noisy_train_indices'):
             raise AttributeError("Provided 'fetcher' object does not have 'noisy_train_indices' attribute.")
        noisy_train_indices = fetcher.noisy_train_indices
    elif data is not None:
        if "x_train" not in data:
            raise KeyError("The 'data' dictionary must contain the key 'x_train'.")
        x_train = data["x_train"]
        # Still need noisy indices, assuming they come from fetcher even if x_train is from data dict
        if fetcher is None or not hasattr(fetcher, 'noisy_train_indices'):
             raise ValueError("When providing 'data', a 'fetcher' with 'noisy_train_indices' is also required.")
        noisy_train_indices = fetcher.noisy_train_indices
    else:
         # This case is covered by the initial check, but added for logical completeness
         raise ValueError("Could not determine source for x_train or noisy_train_indices.")


    if not hasattr(evaluator, 'data_values'):
        raise AttributeError("Provided 'evaluator' object does not have 'data_values' attribute.")
    data_values = evaluator.data_values

    if len(x_train) != len(data_values):
        raise ValueError(f"Length of x_train ({len(x_train)}) must match "
                         f"length of evaluator.data_values ({len(data_values)}).")

    num_points = len(x_train)
    
    # Ensure noisy indices are valid
    if not isinstance(noisy_train_indices, (np.ndarray, list)):
        raise TypeError("noisy_train_indices must be a list or numpy array.")
    noisy_train_indices = np.array(noisy_train_indices) # Ensure numpy array for intersect1d
    num_noisy = len(noisy_train_indices)

    # Sort data values to find lowest valued points
    # Using kind="stable" preserves order for ties, important if indices matter
    sorted_value_indices = np.argsort(data_values, kind="stable")

    # Sort the percentiles to ensure monotonic axis
    sorted_percentiles = sorted(list(set(p for p in percentiles_to_evaluate if 0.0 <= p <= 1.0))) # Ensure unique, valid, sorted

    if not sorted_percentiles:
         print("Warning: No valid percentiles (between 0.0 and 1.0) provided in percentiles_to_evaluate.")
         return {"corrupt_found": [], "axis": []}

    # Handle case with no noisy indices
    if num_noisy == 0:
        print("Warning: No noisy train indices provided or found. Returning 0.0 found rate.")
        found_rates = [0.0] * len(sorted_percentiles)
        eval_results = {"corrupt_found": found_rates, "axis": sorted_percentiles}
        return eval_results

    # Output initialization
    found_rates = []

    # Evaluate at each specified percentile
    for pctl in sorted_percentiles:
        # Determine number of points to consider for this percentile
        # Use round() and ensure it doesn't exceed total points (esp. for pctl=1.0)
        num_points_to_consider = min(int(round(num_points * pctl)), num_points)

        # Get the indices corresponding to the lowest `num_points_to_consider` data values
        indices_at_percentile = sorted_value_indices[:num_points_to_consider]

        # Find intersection between the lowest-value indices and the known noisy indices
        noisy_found_count = len(np.intersect1d(indices_at_percentile, noisy_train_indices, assume_unique=False))
        
        # Calculate the proportion of *all* noisy indices found within this percentile
        found_rate = noisy_found_count / num_noisy
        found_rates.append(found_rate)

    eval_results = {"corrupt_found": found_rates, "axis": sorted_percentiles}

    # Returns True Positive Rate of corrupted label discovery at specified percentiles
    return eval_results


def get_classifier_performance_by_removal(
    evaluator,
    fetcher=None,
    data=None,
    percentages=None,
    remove_low=True,
) -> dict[str, list[float]]:
    """
    Evaluate classifier performance when removing a fraction of training samples
    based on their data valuation. For each removal percentage, the function removes
    the samples with the lowest (or highest, if specified) data values, trains an
    MLP classifier (defined in PyTorch) on the remaining training data, and computes test
    performance.

    Parameters
    ----------
    evaluator : DataEvaluator
        An object that provides data values via its `data_values` attribute.
    fetcher : DataFetcher, optional
        Object containing training and test data attributes:
            - x_train, x_valid, x_test, y_train, y_valid, y_test.
        If not provided, the `data` dict should be used.
    data : dict[str, Any], optional
        Dictionary with the following keys:
            - "x_train", "x_valid", "x_test", "y_train", "y_valid", "y_test".
    percentages : list[float], optional
        List of fractions (between 0 and 1) representing the portion of training data
        to remove. Defaults to [0.1, 0.2, ..., 0.9].
    remove_low : bool, optional
        If True, remove samples with the lowest data values.
        If False, remove samples with the highest data values.

    Returns
    -------
    dict[str, list[float]]
        Dictionary with:
            - "removed_percentage": List of removal percentages.
            - "performance": List of classifier accuracies on the test set after removal.
    """

    # Define device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Retrieve training and test data
    if fetcher is not None:
        x_train = torch_subset_to_tensor(fetcher.x_train)
        y_train = torch_subset_to_tensor(fetcher.y_train)
        x_test  = torch_subset_to_tensor(fetcher.x_test)
        y_test  = torch_subset_to_tensor(fetcher.y_test)
    else:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

    # Convert to torch.Tensor if needed
    if not isinstance(x_train, torch.Tensor):
        x_train = torch.tensor(x_train)
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train)
    if not isinstance(x_test, torch.Tensor):
        x_test = torch.tensor(x_test)
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test)

    # Cast embeddings to float32 (model parameters are float32 by default)
    x_train = x_train.float()
    x_test = x_test.float()

    # Default percentages if none provided
    if percentages is None:
        percentages = [i / 10 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9

    # Sort the indices of training samples based on data values
    sorted_indices = np.argsort(evaluator.data_values, kind="stable")
    if not remove_low:
        # Reverse order to remove highest values
        sorted_indices = sorted_indices[::-1]

    total_points = len(x_train)
    performance_list = []
    removed_percentage_list = []

    # Convert y_test from one-hot to class indices if necessary
    if y_test.dim() > 1 and y_test.size(1) > 1:
        y_test_int = torch.argmax(y_test, dim=1)
    else:
        y_test_int = y_test

    # Loop over each removal percentage
    for p in tqdm(percentages):
        num_to_remove = int(total_points * p)
        # Indices to remove and remaining indices
        indices_to_remove = sorted_indices[:num_to_remove]
        remaining_indices = np.setdiff1d(np.arange(total_points), indices_to_remove)

        # Create reduced training datasets
        x_train_reduced = x_train[remaining_indices]
        y_train_reduced = y_train[remaining_indices]
        
        # Convert one-hot encoded y_train to class indices if needed
        if y_train_reduced.dim() > 1 and y_train_reduced.size(1) > 1:
            y_train_int = torch.argmax(y_train_reduced, dim=1)
        else:
            y_train_int = y_train_reduced

        # Define dataset and dataloader for the reduced training set
        train_dataset = TensorDataset(x_train_reduced, y_train_int)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Define dataloader for the test set
        test_dataset = TensorDataset(x_test, y_test_int)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Determine input dimension and number of classes
        input_dim = x_train_reduced.size(1)
        if y_train_reduced.dim() > 1 and y_train_reduced.size(1) > 1:
            num_classes = y_train_reduced.size(1)
        else:
            num_classes = int(torch.max(y_train_int).item() + 1)
        
        # Define the MLP classifier (similar to the fc part from your ResNet50 code)
        model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        model = model.to(device)

        # Loss function and optimizer (only model parameters are updated)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10  # You can modify this as desired

        # Training loop for the current reduced dataset
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_x.size(0)
            # (Optional) you can track the training loss or accuracy per epoch here

        # Evaluate model performance on the test set
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total_test += batch_x.size(0)
                correct_test += (predicted == batch_y).sum().item()
        test_accuracy = correct_test / total_test

        removed_percentage_list.append(p)
        performance_list.append(test_accuracy)

    return {"removed_percentage": removed_percentage_list, "performance": performance_list}



def est_median_dist(x, n_samples=None, seed=42, per_dim=False):
    """
    Estimates the median distance between pairs of samples from x.
    
    Parameters:
    -----------
    x : np.ndarray
        An array of shape (n, d) where n is the number of samples
        and d is the number of dimensions.
    n_samples : int, optional
        Number of pairs to sample. If None, uses len(x).
    seed : int, default 42
        Random seed for reproducibility.
    per_dim : bool, default False
        If True, return a tuple containing:
            - overall_median: the median of the Euclidean distances
            - median_per_dim: a vector of medians of the absolute differences for each dimension.
        If False, return only the overall median.
    
    Returns:
    --------
    overall_median : float
        The median Euclidean distance between paired samples.
    median_per_dim : np.ndarray (if per_dim is True)
        The median absolute difference computed separately for each dimension.
    """
    if n_samples is None:
        n_samples = len(x)
    np.random.seed(seed)
    
    # Create two shuffled copies
    x1 = x.copy()[np.random.choice(len(x), n_samples, replace=True)]
    x2 = x.copy()[np.random.choice(len(x), n_samples, replace=True)]
    
    # Compute differences
    diffs = x1 - x2
    
    if per_dim:
        # Median absolute difference for each dimension
        median_per_dim = np.median(np.abs(diffs), axis=0)
        return median_per_dim
    else:
        # Overall median Euclidean distance
        euclidean_dists = np.sqrt(np.sum(diffs**2, axis=1))
        overall_median = np.median(euclidean_dists)
        return overall_median


def torch_subset_to_tensor(x):
    if isinstance(x, torch.utils.data.Subset):
        return torch.stack([t for t in x])
        # return torch.tensor(x)
    else:
        return x