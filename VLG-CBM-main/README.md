# VLG-CBM Feature Attribution
Original repository:  
https://github.com/Trustworthy-ML-Lab/VLG-CBM

## Setup Instructions

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and set up CUB. (Not uploaded to GitHub due to size) Refer to instructions in the original repo.

4. Train the model. Refer to instructions in the original repo.

## Running Feature Attribution

The feature attribution can be run using the `attribution.py` script.

### Basic Usage

```bash
python3 attribution.py \
    --model_dirs /path/to/model/checkpoint \
    --dataset your_dataset \
    --output_dir /path/to/output \
    --device cuda  # or cpu
```

### Command Line Arguments

- `--model_dirs`: Path to the model checkpoint file
- `--dataset`: Name of the dataset to use
- `--output_dir`: Directory to save attribution visualizations
- `--device`: Device to run the model on ('cuda' or 'cpu')
- `--model`: Model type ('joint' or 'independent')
- `--use_sigmoid`: Use sigmoid activation for concept predictions (for independent models)
