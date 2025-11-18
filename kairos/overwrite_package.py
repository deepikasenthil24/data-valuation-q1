import os
import numpy as np
import opendataval

def update_register_file():
    # Locate the opendataval package directory.
    package_dir = os.path.dirname(opendataval.__file__)
    file_path = os.path.join(package_dir, "dataloader", "register.py")

    # Read the contents of the file.
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define the original code block we want to replace.
    # Original code:
    #     def one_hot_encode(data: np.ndarray) -> np.ndarray:
    #         data = data.reshape(len(data))  # Reduces shape to (N,) array
    #         num_values = np.max(data) + 1
    #         return np.eye(num_values)[data]
    #
    # We want to modify it to ensure data is a numpy array, for example:
    #     def one_hot_encode(data) -> np.ndarray:
    #         data = np.array(data).reshape(len(data))  # Convert list to array and reduce shape
    #         num_values = np.max(data) + 1
    #         return np.eye(num_values)[data]

    original_line = 'data = data.reshape(len(data))  # Reduces shape to (N,) array'
    new_line = 'data = np.array(data).reshape(len(data))  # Convert list to array and reduce shape'

    # Check if the original line exists in the file.
    if original_line not in content:
        print("The original line was not found in the register file. No changes made.")
        return

    # Replace the original line with the new line.
    updated_content = content.replace(original_line, new_line)

    # Write the modified content back to the file.
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("The register file has been updated successfully.")


def update_datasets_file():
    # Locate the opendataval package directory.
    package_dir = os.path.dirname(opendataval.__file__)
    file_path = os.path.join(package_dir, "dataloader", "datasets", "datasets.py")

    # Read the contents of the file.
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Define the original and new lines.
    original_line = 'category_indices = [dataset["feature_names"].index(x) for x in category_list]'
    new_line = 'category_indices = [dataset["feature_names"].index(x) for x in category_list if x in dataset["feature_names"]]'

    # Check if the original line exists in the file.
    if original_line not in content:
        print("The original line was not found in the datasets file. No changes made.")
        return

    # Replace the original line with the new line.
    updated_content = content.replace(original_line, new_line)

    # Write the modified content back to the file.
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("The datasets file has been updated successfully.")

    
def update_fetcher_file_all():
    # Locate the opendataval package directory.
    package_dir = os.path.dirname(opendataval.__file__)
    file_path = os.path.join(package_dir, "dataloader", "fetcher.py")

    # Read the contents of the file.
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ensure that NumPy is imported; if not, add it at the top.
    if "import numpy as np" not in content:
        content = "import numpy as np\n" + content

    # Define original and new lines for train, validation, and test splits.
    replacements = {
        'x_trn = torch.tensor(x_trn, dtype=torch.float).view(-1, *self.covar_dim)': 
        'x_trn = torch.tensor(np.array(x_trn), dtype=torch.float).view(-1, *self.covar_dim)',
        
        'x_val = torch.tensor(x_val, dtype=torch.float).view(-1, *self.covar_dim)': 
        'x_val = torch.tensor(np.array(x_val), dtype=torch.float).view(-1, *self.covar_dim)',
        
        'x_test = torch.tensor(x_test, dtype=torch.float).view(-1, *self.covar_dim)': 
        'x_test = torch.tensor(np.array(x_test), dtype=torch.float).view(-1, *self.covar_dim)',
    }

    updates = 0
    
    # Replace each original line with its corresponding new line.
    for original_line, new_line in replacements.items():
        if original_line in content:
            content = content.replace(original_line, new_line)
            updates += 1
        # else:
        #     print(f"Original line not found: {original_line}")
    
    if updates == 0:
        print("The original lines were not found in the fetcher file. No changes made.")
    else:
        # Write the updated content back to the file.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("The fetcher file has been updated successfully.")

    
def update_noisify_file():
    # Locate the opendataval package directory.
    package_dir = os.path.dirname(opendataval.__file__)
    file_path = os.path.join(package_dir, "dataloader", "noisify.py")

    # Replace the old noisify with the new one
    os.system(f'cp new_noisify.py {file_path}')

    print("The noisify.py file has been replaced successfully.")

def update_imagesets_file():
    # Locate the opendataval package directory.
    package_dir = os.path.dirname(opendataval.__file__)
    file_path = os.path.join(package_dir, "dataloader", "datasets", "imagesets.py")

    # Replace the old noisify with the new one
    os.system(f'cp new_imagesets.py {file_path}')

    print("The imagesets.py file has been replaced successfully.")

def update_nlpsets_file():
    # Locate the opendataval package directory.
    package_dir = os.path.dirname(opendataval.__file__)
    file_path = os.path.join(package_dir, "dataloader", "datasets", "nlpsets.py")

    # Replace the old noisify with the new one
    os.system(f'cp new_nlpsets.py {file_path}')

    print("The nlpsets.py file has been replaced successfully.")

def update_dataloader_util_file():
    package_dir = os.path.dirname(opendataval.__file__)
    file_path = os.path.join(package_dir, "dataloader", "util.py")

    # Replace the old noisify with the new one
    os.system(f'cp new_dataloader_util.py {file_path}')

    print("The dataloader.util has been replaced successfully.")

if __name__ == "__main__":
    update_noisify_file()
    update_imagesets_file()
    update_nlpsets_file()
    update_fetcher_file_all()
    update_datasets_file()
    update_register_file()
    update_dataloader_util_file()
