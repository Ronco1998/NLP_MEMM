# HW1: NLP Sequence Tagging with MEMM

## Overview
This project implements a sequence tagging system for Natural Language Processing (NLP) tasks using a Maximum Entropy Markov Model (MEMM). The system is designed to extract features from text, train MEMM models, and perform part-of-speech (POS) tagging or similar sequence labeling tasks. The project includes scripts for training, inference, evaluation, and competition submission.

## Directory Structure
- `code/` — Main source code for feature extraction, training, optimization, and inference.
  - `main.py` — Main entry point for training and evaluation.
  - `preprocessing.py` — Feature extraction and data preprocessing.
  - `optimization.py` — Model training and optimization routines.
  - `inference.py` — Inference and tagging logic.
- `data/` — Training, test, and competition data files.
- `comps files/` — Output directory for competition files.
- `check_submission.py` — Script to validate and package your submission.
- `generate_comp_tagged.py` — Script to generate competition predictions.
- `tag_comp.py` — Baseline POS tagger using NLTK.
- `confusion_matrix_comps.py` — Script for confusion matrix analysis.

## Requirements
- Python 3.8+
- Required packages: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`, `tqdm`

Install dependencies with:
```powershell
pip install numpy scipy pandas scikit-learn matplotlib seaborn nltk tqdm
```

## Usage
### 1. Training and Evaluation
Run the main script to train models and generate predictions:
```powershell
python code/main.py
```
This will train MEMM models on the provided training data and output predictions for the test and competition sets.

### 2. Generating Competition Files
To generate the `.wtag` files for competition submission:
```powershell
python generate_comp_tagged.py
```

### 3. Baseline Tagging
To generate baseline tags using NLTK:
```powershell
python tag_comp.py
```

### 4. Submission Check
To validate your submission package:
```powershell
python check_submission.py
```
Follow the prompts to enter your student IDs and check the required files.

## Data
- Place your training and test files in the `data/` directory.
- Output predictions and competition files will be saved in the project root or `comps files/` as specified.

## Feature Engineering
The system extracts a variety of features, including:
- Word/tag pairs
- Prefixes and suffixes
- Previous/next words and tags
- Capitalization, hyphens, numbers, domain-specific terms (biology/economics), etc.

See `code/preprocessing.py` for full details on feature extraction.

## Report
See `report_*.pdf` for a detailed description of the approach, experiments, and results.

## License
This project is for academic use only.
