# Data Valuation Quarter 1 Project for DSC180AB

This repository hosts implementations of various data valuation frameworks (**LAVA** and **KAIROS**) and a framework for identifying worst-case data corruption patterns (**SAVAGE**).

***

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
Adult Dataset (Small Sample) |	"dataset": "adult", "sample_frac": 0.05
Wine Dataset (Full Sample) |	"dataset": "wine", "sample_frac": 1.0

#### 2. Changing Corruption Type
The SAVAGE beam search can be configured to identify the worst-case pattern for five different types of data corruption by setting the `error_type` parameter in the `script.py` function call.

| Corruption Type | `error_type` Value | Impact |
| :--- | :--- | :--- |
| **Feature Missingness** | `'MNAR'` (Default) | Replaces feature values in a column of x_train with NAN.|
| **Label Errors** | `'Label'` | Flips (relabels) the class label in y_train. |
| **Selection Bias** | `'Sampling'` | Drops entire rows from the training data. |
| **Feature Outliers** | `'OutlierError'` | Multiplies numerical feature values by a factor (default 1.5). |
| **Data Duplication** | `'DuplicateError'` | Duplicates and appends identified rows to the training data. |

```python
# In savage/script.py
top_results_auc = run_beam_search(
    # ... (7 positional arguments) ...
    error_type='OutlierError', # Change to 'Label', 'Sampling', 'DuplicateError', etc.
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

## 2. LAVA
coming soon

## 3. KAIROS
coming soon
