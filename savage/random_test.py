import pandas as pd
import matplotlib.pyplot as plt
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

with open('experiment_results_employee.json', 'r') as file:
    results = json.load(file)

data = {
    "error_col": [],
    "auc_value": [],
    "pattern_cols": []
}

for result in results:
    for key, value in result.items():
        if key == "pattern_cols":

            data["pattern_cols"].append(", ".join(value))
        else:
            data["error_col"].append(key)
            data["auc_value"].append(value)

df = pd.DataFrame(data)

plt.figure(figsize=(12, 8))
plt.hist(df['auc_value'], bins=20, color='blue', alpha=0.7)
plt.title('AUC Distribution')
plt.xlabel('AUC Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

min_auc = df['auc_value'].min()
worst_experiment = df[df['auc_value'] == min_auc]

worst_ten_experiments = df.nsmallest(10, 'auc_value')

print("Worst 10 Experiments Details:")
print(worst_ten_experiments[['error_col', 'pattern_cols', 'auc_value']])

plt.figure(figsize=(12, 8))
plt.scatter(df.index, df['auc_value'], color='blue', alpha=0.5)
plt.scatter(worst_experiment.index, worst_experiment['auc_value'], color='red')
plt.title('AUC Scores for Experiments')
plt.xlabel('Experiment Index')
plt.ylabel('AUC Score')
plt.grid(True)
plt.show()