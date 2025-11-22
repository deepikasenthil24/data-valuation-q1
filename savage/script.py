import warnings
import json
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from load_dataset import load
from savage import run_beam_search
from code import pipeline, auc

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Exiting.")
    exit()

dataset = config['dataset']
budget_pct = config['budget_pct']
top_k = config['top_k']
RANDOM_STATE = config['random_state']
SAMPLE_FRAC = config['sample_frac']

warnings.filterwarnings('ignore')
print(f"--- Experiment Setup ---\nDataset: {dataset}\n")

# Load and Split data
X_train, X_test, y_train, y_test = load(dataset)

# Split test data further for validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

# Use sampling for efficiency
X_train = X_train.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
X_test = X_test.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
y_train = y_train.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
y_test = y_test.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
X_val = X_val.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
y_val = y_val.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

# Reset indices and create copies
X_train_orig = copy.deepcopy(X_train).reset_index(drop=True)
X_test_orig = copy.deepcopy(X_test).reset_index(drop=True)
X_val_orig = copy.deepcopy(X_val).reset_index(drop=True)
y_train, y_test, y_val = y_train.reset_index(drop=True), y_test.reset_index(drop=True), y_val.reset_index(drop=True)

# Calculate Clean Metrics
print("--- Calculating Clean Metrics ---")
y_pred_clean = pipeline(X_train_orig, y_train, X_test_orig)
clean_auc = auc(X_test_orig, y_test, y_pred_clean)
print(f'Clean AUC: {clean_auc:.4f}')
print("---------------------------------\n") 

# Set up SAVAGE parameters
budget = int(X_train_orig.shape[0] * budget_pct)
print(f"Total budget for missing data: {budget} rows ({budget_pct*100}%)")

# --- Run SAVAGE for AUC (Utility) ---
print("--- Start SAVAGE Beam Search for AUC (Utility) ---")

#top_results_auc = run_beam_search(X_train_orig, X_test_orig, y_train, y_test, pipeline, auc, budget, top_k=top_k)

top_results_auc = run_beam_search(
    X_train_orig,                        
    X_val_orig,                        
    y_train,                             
    y_val,                     
    pipeline,                           
    lambda X, y, y_pred: auc(X, y, y_pred), 
    budget,                         
    error_type='Label',             
    random_state=RANDOM_STATE,        
    top_k=top_k                    
)


budget = int(X_train_orig.shape[0] * budget_pct)

if top_results_auc:
    r_auc = top_results_auc[0]
    best_pattern = r_auc[0]
    
    X_train_dirty = r_auc[1][1]
    y_train_dirty = r_auc[1][2]
    
    print("\n--- Saving Data Splits ---")
    
    X_train_dirty.to_csv("X_train_dirty.csv", index=False)
    y_train_dirty.to_csv("y_train_dirty.csv", index=False)
    
    X_train_orig.to_csv("X_train_clean.csv", index=False)
    y_train.to_csv("y_train_clean.csv", index=False)
    
    X_val_orig.to_csv("X_val.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)
    
    X_test_orig.to_csv("X_test.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    
    print("All data splits saved successfully.")
    
    auc_drop = clean_auc - r_auc[1][0]
    print("\n--- Worst-Case AUC Harm Result ---")
    print(f"Change of data in column {best_pattern[0]} depending on pattern {best_pattern} could lead to an AUC drop of {auc_drop:.4f}")
    print(f'Worst-case AUC found: {r_auc[1][0]:.4f}')
    print("----------------------------------\n")