# Kairos
Implementation for paper: KAIROS: Scalable Model-Agnostic Data Valuation

## Environment Setup

Set up your development environment by running the following commands:

```bash
# Step 1: Install required packages from requirements.txt
python3 -m pip install -r requirements.txt

# Step 2: Install the benchmark tool opendataval
python3 -m pip install --no-dependencies opendataval

# Step 3: Fix the data-loading bug in the package by running the overwrite script
python3 overwrite_package.py
```
## Newest Results
- Image classification datasets (CIFAR10, STL10, SVHN): `image-data.ipynb`.
- Text classification datasets (IMDB, AGNews): `text-data.ipynb`.

## Modifications to the original Opendataval package (details in `overwrite_package.py`)
- Fix deprecated usages and data loading issues.
- Keep the validation set free from corruption.

## Download Precomputed Embeddings
- Download precomputed embeddings from [here](https://drive.google.com/file/d/1JXOG4_zyDCjlSQf8phQ2jXQ5Hh8I5Pjt/view?usp=sharing).
- Put the unzipped directory as `data_files/`.
