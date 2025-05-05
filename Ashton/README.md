# Joint Model of [Concept Bottleneck Models](https://github.com/yewsiang/ConceptBottleneck)

## Steps to run `feature_attribution.py` and generate heatmaps:

1. Download complete raw CUB dataset, ensure no files/subfolders are missing from dataset
    - The `CUB_processed_pkl` directory contains all the pkl files necessary for `feature_attribution.py` to work; no need to preprocess downloaded CUB dataset
2. Run `pip install --user -r requirements.txt` to download all the necessary dependencies
3. Run `feature_attribution.py` by following this setup: `python3 feature_attribution.py -model joint -model_dirs outputs/best_model_1.pth -outputDir ./processed-photos -use_attr`
    - `-model joint` specifies that the joint model is the primary model type being used here
    - `-model_dirs ./pretrained_models/MODEL.pth` specifies that `MODEL.pth` is the .pth model used to generate the heatmmaps. 
        - Ensure that you have a subdirectory that contains a joint-trained .pth file, which could be a pretrained model from the *ConceptBottleneck* scienfic paper, or use their github repo to train your own model
    - `-outputDir ./processed_photos` specifies that any generated heatmap .jpg files will be downloaded into the ./processed-photos directory
    - `-use_attr` specifies whether attributes will be used
4. Running `feature_attribution.py` will begin at the first bird class (in `classes.txt`) and run down the list. You may modify the commented out lines of code (lines 212-214) if you choose to begin processing other bird species first

Note: `feature_attribution.py` will skip certain bird photos if it (1.) misidentifies the bird species, or (2.) can't confidentally recognize any attributes/concepts in a photo

Note: `feat_attr.sh` was the respective bash file to run `feature_attribution.py` on USC CARC servers. You can still refer to that bash file to understand how testing was done.
