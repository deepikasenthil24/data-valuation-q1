# Data Valuation Quarter 1 Project for DSC180AB

This repository hosts implementations of two data valuation frameworks (**LAVA** and **KAIROS**) and a framework for identifying worst-case data corruption patterns (**SAVAGE**).


## 1. SAVAGE: Identifying Worst-Case Data Corruption (AUC)

The **SAVAGE** (Sensitivity Analysis Via Automatic Generation of Errors) pipeline is configured to find the pattern (subset of data points and features) that, when corrupted (i.e., data masked), causes the maximum drop in model Utility (AUC). The output of the search is a complete set of clean and corrupted data splits for analysis.

### A. Prerequisites

The script requires a standard Python environment with the following libraries:

```bash
pip install numpy pandas scikit-learn optuna
```


### B. Project Structure
The files for this method are self-contained within the savage/ subdirectory:
```
└── savage/
    ├── code.py           # ML pipeline (Logistic Regression) and AUC metric definition
    ├── config.json       # Experiment configuration (defines budget, dataset, etc.)
    └── script.py         # Main execution file
    ├── load_dataset.py   # Data loading utility
    └── savage.py         # Core SAVAGE beam search implementation
```

### C. Running the Script to Generate Injected Data
1. Navigate to the SAVAGE Directory:
```bash
cd savage
```

2. Execute the Script:
```bash
python script.py
```

### D. Customizations
#### 1. Changing Datasets (Adult vs. Wine)
Experiment configurations are managed entirely through savage/config.json. To switch between datasets and adjust sampling fractions, edit the dataset and sample_frac fields:
Experiment	| config.json Setting
| :--- | :--- |
Adult Dataset |	"dataset": "adult", "sample_frac": 0.05

#### 2. Changing Corruption Type
The SAVAGE beam search can be configured to identify the worst-case pattern for four different types of data corruption by setting the `error_type` parameter in the `script.py` function call.

| Corruption Type | `error_type` Value | Impact |
| :--- | :--- | :--- |
| **Feature Missingness** | `'MNAR'` (Default) | Replaces feature values in a column of x_train with NAN.|
| **Label Errors** | `'Label'` | Flips (relabels) the class label in y_train. |
| **Feature Outliers** | `'Outlier'` | Multiplies numerical feature values by a factor (default 1.5). |
| **Data Duplication** | `'Duplicate'` | Duplicates and appends identified rows to the training data. |

```python
# In savage/script.py
top_results_auc = run_beam_search(
    # ... (7 positional arguments) ...
    error_type='Outlier', # Change to 'Label', 'Duplicate', etc.
    random_state=RANDOM_STATE,
    top_k=top_k
)
```
   

### E. Output: Saved Data Split
Upon successful completion of the beam search, the script will generate eight CSV files in the savage/ directory, saving the complete set of clean data splits and the identified worst-case corrupted training data.

| Output File Name | Description |
| :--- | :--- |
| `X_train_dirty.csv`| The worst-case training feature matrix with injected missing data. |
| `y_train_dirty.csv` | The labels corresponding to the dirty training feature matrix. |
| `X_train_clean.csv` | The clean (original) training feature matrix. |
| `y_train_clean.csv` | The clean (original) training labels. |
| `X_val.csv` | The clean validation feature matrix. |
| `y_val.csv` | The clean validation labels. |
| `X_test.csv` | The clean test feature matrix. |
| `y_test.csv` | The clean test labels. |

The console output will display the maximum AUC drop found and confirm that the data splits were successfully saved.

## 2. Main Experiment Runner & KAIROS

### A. Prerequisites

The script requires a Python *3.9* environment

```bash
# Move to the Kairos directory to run the relevant files
cd kairos

# Step 1: Install required packages from requirements.txt
python3 -m pip install -r requirements.txt

# Step 2: Install the benchmark tool opendataval
python3 -m pip install --no-dependencies opendataval

# Step 3: Fix the data-loading bug in the package by running the overwrite script
python3 overwrite_package.py
```
Then you should be able to run the code in experiment_runner.ipynb


### B. Project Structure
The files for this method are self-contained within the kairos/ subdirectory:
```
└── kairos/
    ├── otdd/                   # Optimal transport dataset distance
        ├── pytorch/            # distance functions
        ├── plotting.py         # plotting funtions
        └── utils.py            # supporting functions
    ├── README.md               # Kairos specific documentation
    ├── custom_valuations.py    # Kairos class
    ├── fixed_valuations.py     # other methods classes (Lava, KNN Shapley)
    ├── new_dataloader_util.py  # loaders for various datasets
    ├── overwrite_package.py    # overwrites bug in opendataval
    └── requirements.txt        # dependencies

***

### Experiment Results
| Experiment Number | Description | LAVA AUC | LAVA Time | KAIROS AUC | KAIROS Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Missing feature value error injection | 0.99 | 2.41 | 0.79 | 1.23 |
| 2 | Label error injection | 0.50 | 1.81 | 0.83 | 1.06 |
| 3 | Feature outlier injection | 1.0 | 2.35 | 1.0 | 1.08 |
| 4 | Duplicate injection | .81 | 1.91 | 0.82 | 0.60 |

***

 

```
