import itertools
import subprocess
import json

columns = ["Education", "JoiningYear", "PaymentTier", "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain",
           "City_Bangalore", "City_New Delhi", "City_Pune"]
error_pct = 0.5

results = []

def run_experiment(error_col, pattern_cols):
    command = [
        'python', 'test.py',
        '--dataset', 'employee',
        '--cleaning', 'h2o',
        '--model', 'LR',
        '--error_type', 'MNAR',
        '--error_pct', str(error_pct),
        '--error_cols', error_col,
        '--pattern_cols', *pattern_cols,
        '--sens_attr', 'Gender',
        '--objective', 'AUC',
        '--n_trials', '100',
        '--n_processes', '10',
        '--override'
    ]
    subprocess.run(command)

    result_filename = f"employee_results_h2o_MNAR_AUC_LR.json"
    with open(result_filename, 'r') as file:
        data = json.load(file)
        auc_value = data[error_col]["AUC"]["means"][1]
    return {error_col: auc_value, 'pattern_cols': pattern_cols}

for error_col in columns:
    # 包括 'Y' 在可能的 pattern_cols 特征中
    all_pattern_columns = columns + ['Y']
    # 对于每个 error_col，使用一个集合来避免重复的 pattern_cols
    processed_patterns = set()
    # 从 0 到 4 个特征逐渐增加
    for num_pattern_cols in range(0, 5):
        # 从 all_pattern_columns 中选择 num_pattern_cols 个特征
        for combo in itertools.combinations(all_pattern_columns, num_pattern_cols):
            if num_pattern_cols > 0 and error_col not in combo:
                continue
            # 将组合转换为 frozenset 以确保顺序无关
            pattern_cols_set = frozenset(combo)
            if pattern_cols_set in processed_patterns:
                continue
            processed_patterns.add(pattern_cols_set)
            pattern_cols = list(combo)
            result = run_experiment(error_col, pattern_cols)
            results.append(result)
            print(f"Completed: Error Column: {error_col}, Pattern Columns: {pattern_cols}, Result: {result}")

with open('experiment_results_employee.json', 'w') as f:
    json.dump(results, f, indent=4)

print(results)