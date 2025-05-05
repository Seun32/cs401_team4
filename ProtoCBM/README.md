## ------------------------- Documentation for CSCI401 ---------------------------------- ##

In the beginning of the semester the group was tasked to researched concept bottleneck models (CBM) and I came across the article "Onthe Concept Trustworthiness in Concept Bottleneck Models" that talks about allocating a trustworthiness score for the concept with the model ProtoCBM.

After learning about integrated gradients and pixel attribution I trained the model on the CUB2011 birds dataset in Google Colab/CARC following instructions from the given GitHub repo (https://github.com/hqhQAQ/ProtoCBM).

I then used SheltonZhaoK's (GitHub) feature_attribution.py to visualize how the model is interpreting the images for Integrated Gradients and Integrated Gradient Smoothed.

A similar code was used for the heatmap, which was then run through ChatGPT, prompting it to ask yes or no questions to see if it could interpret if the pixels were on a portion of the bird's body.

## ------------------------- Instructions for CSCI401 ---------------------------------- ##

The code was kept the same and the only changes made were the addition of feature_attribution.py from SheltonZhaoK
(GitHub) and creating a heatmap using similar code.

Instructions:

Follow the steps in "Documentation before CSCI401". (Some requirements were deprecated since then, a new requirements.txt is provided but may not be reliable)

Once epoch-last.pth is retrieved (check output_cosine/CUB2011/resnet18/1028-1e-4-adam-18-train/checkpoints) run feature_attribution.py using 'python3 feature_attribution.py'

Run 'python heatmap.py' to get the highlight green pixels overlayed on the image.

Import the image into your chosen AI chatbot and prompt it with a question in the format:

"Are the [insert feature] of the bird highlighted by the green color? By highlighting I mean most part of the targeted position is covered. Only answer yes or no."

## ---------------------- Documentation before CSCI401 hqhQAQ (GitHub) -------------------- ##
# Trustworthy CBM

## Introduction

This is the official implementation of paper **''Onthe Concept Trustworthiness in Concept Bottleneck Models''**.

## Required Packages

Our required python packages are listed as below:

```
pytorch==1.12.1+cu113
Augmentor==0.2.9
torchvision==0.13.1+cu113
pillow==8.4.0
timm==0.5.4
opencv-python==4.6.0.66
tensorboard==2.9.1
scipy==1.8.1
pandas==1.4.3
matplotlib==3.5.2
scikit-learn==1.1.1
pytorchcv==0.0.67
```

## Dataset Preparation

* Download the dataset (CUB_200_2011.tgz) from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
* Unpack CUB_200_2011.tgz to the `datasets/` directory in this project (the path of CUB-200-2011 dataset will be `datasets/CUB_200_2011/`).
* Download the concept annotations (class_attr_data_10) from https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683, which is from the vanilla CBM. Next, put it into the `datasets/` directory in this project (the path of `class_attr_data_10` will be `datasets/class_attr_data_10/`)
* Run `python util/crop_cub_data.py` to split the cropped images into training and test sets. The cropped training images will be in the directory `datasets/cub200_cropped/train_cropped/`, and the cropped test images will be in the directory `datasets/cub200_cropped/test_cropped/`.
* Run `python util/img_aug.py --data_path /path/to/source_codes/datasets/cub200_cropped` to augment the training set. Note that `/path/to/source_codes/datasets/cub200_cropped` should be an absolute path. This will create an augmented training set in the following directory: `datasets/cub200_cropped/train_cropped_augmented/`.

## Training Instructions

Use `scripts/train.sh` for training:

```
sh scripts/train.sh $model $num_gpus
```

Here, `$model` is the name of backbone chosen from `resnet18, resnet34, resnet152, densenet121, densenet161, deit_tiny, deit_small, swin_small`. `num_gpus` is the number of GPUs. Note that our model is trained with 2 GPUs.

For example, the instruction for training a ResNet18 model with 2 GPUs is as below:

```
sh scripts/train.sh resnet18 2
```

## Evaluate the Concept Trustworthiness Score

The instruction for evaluating the concept trustworthiness score of a ResNet18 model with checkpoint path `$ckpt_path`:

```
python eval_concept_trustworthiness.py \
--base_architecture resnet18 \
--resume $ckpt_path
```