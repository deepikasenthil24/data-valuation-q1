import itertools
import subprocess
import json

columns = ["Education", "JoiningYear", "PaymentTier", "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain",
           "City_Bangalore", "City_New Delhi", "City_Pune"]
# columns = ["route", "operator", "group_name", "bus_garage", "borough", "incident_event_type",
#            "victim_category", "victims_sex", "victims_age"]
# columns = ["decile1b", "decile3", "lsat", "ugpa", "zfygpa", "zgpa", "fulltime", "fam_inc", "male", "racetxt", "tier"]
error_pct = 0.5

results = []


def run_experiment(error_col, pattern_cols):
    command = [
        'python', 'test.py',
        '--dataset', 'employee',
        # '--dataset', 'tfl',
        # '--dataset', 'law',
        '--cleaning', 'h2o',
        '--model', 'LR',
        '--error_type', 'MNAR',
        '--error_pct', str(error_pct),
        '--error_cols', error_col,
        '--pattern_cols', *pattern_cols,
        '--sens_attr', 'Gender',
        # '--sens_attr', 'victims_sex',
        # '--sens_attr', 'male',
        '--objective', 'AUC',
        '--n_trials', '100',
        '--n_processes', '10',
        '--override'
    ]
    subprocess.run(command)

    result_filename = f"employee_results_h2o_MNAR_AUC_LR.json"
    # result_filename = "tfl_results_h2o_MNAR_AUC_LR.json"
    # result_filename = "law_results_h2o_MNAR_AUC_LR.json"
    with open(result_filename, 'r') as file:
        data = json.load(file)
        auc_value = data[error_col]["AUC"]["means"][1]
    return {error_col: auc_value, 'pattern_cols': pattern_cols}


for error_col in columns:
    other_cols = [col for col in columns if col != error_col]
    for combo in itertools.combinations(other_cols, 2):
        pattern_cols = [error_col] + list(combo) + ['Y']
        result = run_experiment(error_col, pattern_cols)
        results.append(result)
        print(f"Completed: Error Column: {error_col}, Pattern Columns: {pattern_cols}, Result: {result}")

with open('experiment_results_employee.json', 'w') as f:
    json.dump(results, f, indent=4)

print(results)