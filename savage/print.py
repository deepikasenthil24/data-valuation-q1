import json
import numpy as np
import matplotlib.pyplot as plt

def plot_auc_comparison(json_files, labels, output_dir):
    soft_colors = ['#7fbf7b', '#af8dc3', '#fdae61', '#67a9cf', '#d7191c']

    all_data = {attr: {label: None for label in labels} for attr in ['education', 'marital', 'gender']}

    for json_path, label in zip(json_files, labels):
        with open(json_path, 'r') as f:
            results = json.load(f)
        for attr in all_data.keys():
            all_data[attr][label] = results[attr]['AUC']

    for attr, methods_data in all_data.items():
        plt.figure(figsize=(12, 8))

        example_label = next(iter(methods_data))
        budgets = np.array(methods_data[example_label]['budgets'])
        positions = np.arange(len(budgets))

        for i, (label, auc_data) in enumerate(methods_data.items()):
            means = np.array(auc_data['means'])
            stds = np.array(auc_data['stds'])
            pos_offset = positions + (i - len(labels) / 2) * 0.15

            plt.errorbar(pos_offset, means, yerr=stds, fmt='o', color=soft_colors[i % len(soft_colors)],
                         ecolor='black', capsize=5, elinewidth=2, markeredgewidth=2, label=label)

        plt.title(f'AUC vs Number of NaNs for {attr.capitalize()}')
        plt.xlabel('Number of NaNs')
        plt.ylabel('AUC')
        plt.xticks(positions, budgets)
        plt.legend()
        plt.grid(True, axis='y')
        plt.ylim(0.7, 1)
        output_path = f"{output_dir}/AUC_comparison_{attr}.pdf"
        plt.savefig(output_path, format='pdf', dpi=200)
        plt.close()
        print(f"Saved: {output_path}")

json_files = [
    "final_results_boostclean_Missing.json",
    "final_results_diffprep_Missing.json",
    "final_results_diffprep_random_Missing_MNAR.json",
    "final_results_learn2clean_Missing.json"
]

labels = ["BoostClean", "DiffPrep", "DiffPrep_Random", "Learn2Clean"]

output_dir = "/Users/albertxu/data-err-experiment/Results"

plot_auc_comparison(json_files, labels, output_dir)
