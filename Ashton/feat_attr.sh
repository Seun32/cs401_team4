#!/bin/bash
#SBATCH --job-name=attr_job
#SBATCH --output=attr_output.log
#SBATCH --error=attr_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --time=02:00:00

export PYTHONPATH=$PYTHONPATH:/home1/ashtonch/deliverable5/CUB

pip install --user -r requirements.txt

# Optional: reduce OpenBLAS threads to avoid errors
export OPENBLAS_NUM_THREADS=1

# Activate your virtualenv or conda env here if needed
# source ~/myenv/bin/activate

# Run the script
python3 feature_attribution.py -model joint -model_dirs ./pretrained_models/MODEL.pth -outputDir ./processed-photos -use_attr

