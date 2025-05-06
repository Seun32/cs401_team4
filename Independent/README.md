# Concept Bottleneck Model (Independent) - CUB Dataset
## Dataset preprocessing
Download the official CUB dataset (CUB_200_2011)

Run data_processing.py to obtain train/ val/ test splits as well as to extract all relevant task and concept metadata into pickle files.

## Experiments
Update the paths (e.g. BASE_DIR, -log_dir, -out_dir, --model_path) in the scripts (where applicable) to point to your dataset and outputs.
Run the scripts below to get the results for 1 seed. Change the seed values and corresponding file names to get results for more seeds.

Train the x -> c model:

python3 src/experiments.py cub Concept_XtoC --seed 1 -ckpt 1 -log_dir ConceptModel__Seed1/outputs/ -e 1000 -optimizer sgd -pretrained -use_aux -use_attr -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 1000 -bottleneck

Train the c -> y model:
python3 src/experiments.py cub Independent_CtoY --seed 1 -log_dir IndependentModel_WithVal___Seed1/outputs/ -e 500 -optimizer sgd -use_attr -data_dir CUB_processed/class_attr_data_10 -n_attributes 112 -no_img -b 64 -weight_decay 0.00005 -lr 0.001 -scheduler_step 1000 


## Feature attribution 
Use feature_attribution.py

python3 CUB/feature_attribution.py -model independent -model_dirs CUB/outputs/conceptModel/best_model_1.pt -model_dirs2 CUB/outputs/labelModel/best_model_1.pt -outputDir /home/konghaoz/cbm/CUB/outputs/conceptModel/ensemble -use_sigmoid -use_attr


## Prerequisites
please run `pip install -r requirements.txt`. Main packages are:
- matplotlib 3.1.1
- numpy 1.17.1
- pandas 0.25.1
- Pillow 6.2.2
- scipy 1.3.1
- scikit-learn 0.21.3
- torch 1.1.0
- torchvision 0.4.0




