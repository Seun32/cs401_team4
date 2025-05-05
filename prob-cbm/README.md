# Probabilistic Concept Bottleneck Models

Link to original Probabilistic Concept Bottleneck Model code: https://github.com/ejkim47/prob-cbm
Link to research paper: https://arxiv.org/pdf/2306.01574

## Use

Download official CUB dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/
Download Processed CUB dataset: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683

For current configiration:
Make a datasets folder and place the attributes.txt, CUB_200_2011 folder, and class_attr_data_10 folder inside.

You will also need to connect a wandb account before running the model

Run:
```bash
python main.py --config ./configs/config_exp.yaml --gpu {gpu_num}
```
or 
```bash
python main.py --config ./configs/config_base.yaml --gpu {gpu_num}
```

## Feature Attribution

Was unsuccessful in producing feature attribution for this model