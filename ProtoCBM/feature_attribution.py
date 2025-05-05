import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import transforms

import model
from util.datasets import Cub2011AttributeWhole
from util.preprocess import mean, std
from captum.attr import IntegratedGradients, NoiseTunnel
import pandas as pd

class ConceptPredictor(torch.nn.Module):
    def __init__(self, model1):
        super(ConceptPredictor, self).__init__()
        self.conceptModel = model1

    def forward(self, x):
        (_, _, attributes_logits), _ = self.conceptModel(x)
        return attributes_logits
        
class LabelPredictor(torch.nn.Module):
    def __init__(self, model, args):
        super(LabelPredictor, self).__init__()
        self.model = model
        self.args = args
        
    def forward(self, x):
        (_, logits_attri, _), _ = self.model(x)
        
        return logits_attri

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    ppnet = model.construct_CBMNet(
        base_architecture='resnet18',  
        pretrained=False,
        img_size=224,
        prototype_shape=[2000, 64, 1, 1], 
        num_classes=200,
        prototype_activation_function='log',
        add_on_layers_type='regular'
    )
    
    ppnet.load_state_dict(checkpoint['model'], strict=False)
    ppnet.eval()
    
    return ppnet

def ig_attribution(model, inputs, target_index, return_convergence_delta=True):
    inputs.requires_grad_()
    ig = IntegratedGradients(model)
    attribution, _ = ig.attribute(inputs, target = target_index, return_convergence_delta=return_convergence_delta)
    return attribution

def ig_attribution_smooth(model, inputs, target_index):
    inputs.requires_grad_()
    ig = IntegratedGradients(model)
    smooth_ig = NoiseTunnel(ig)
    attribution = smooth_ig.attribute(
        inputs, target=target_index, nt_type='smoothgrad', stdevs=0.1, nt_samples=5)
    return attribution

def visualize_attribution_new(inputs, label_attributions_list, concept_attributions_list, concepts_names, concept_predictions, className, predictClass, path=None):
    original_image = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    axes_id = 0
    total_images = 1 + len(label_attributions_list)*4
    fig, axes = plt.subplots(1, total_images, figsize=(10+5*7, 5))

    # Plot original image
    axes[axes_id].imshow(original_image)
    axes[axes_id].axis('off')
    axes[axes_id].set_title(f'{className}')
    axes_id += 1

    label_threshold = 0.05
    concept_threshold = 0.001
    for label_attributions, concept_attributions, method_name in zip(label_attributions_list, concept_attributions_list, ["IG", "smoothed IG"]):
        label_attribution_image = label_attributions.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
        label_attribution_image = (label_attribution_image - label_attribution_image.min()) / (label_attribution_image.max() - label_attribution_image.min())
        label_threshold_value = np.percentile(label_attribution_image, (1 - label_threshold) * 100)
        label_attribution_image[label_attribution_image < label_threshold_value] = 0
        label_attribution_image[label_attribution_image >= label_threshold_value] = 1
        new_image = np.ones((label_attribution_image.shape[:2][0], label_attribution_image.shape[:2][1], 3))
        boolean_mapping = [(label_attribution_image == 1)[:, :, i] for i in range(3)]

        boolean_mapping_reverse = [(label_attribution_image == 0)[:, :, i] for i in range(3)]

        for channel in boolean_mapping_reverse:
            new_image[channel, 2] -= 0.015

        for channel in boolean_mapping:
            new_image[channel, 2] -= 0.33
            new_image[channel, 1] -= 0.33
        label_attribution_image = new_image

        processed_concept_attributions = []
        for concept_attribution in concept_attributions:
            concept_image = concept_attribution.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
            concept_image = (concept_image - concept_image.min()) / (concept_image.max() - concept_image.min())
            concept_threshold_value = np.percentile(concept_image, (1 - concept_threshold) * 100)  # Top p
            concept_image[concept_image < concept_threshold_value] = 0
            concept_image[concept_image >= concept_threshold_value] = 1

            new_image = np.ones((concept_image.shape[:2][0], concept_image.shape[:2][1], 3))
            boolean_mapping = [(concept_image == 1)[:, :, i] for i in range(3)]
            boolean_mapping_reverse = [(concept_image == 0)[:, :, i] for i in range(3)]

            for channel in boolean_mapping_reverse:
                new_image[channel, 2] -= 0.015

            for channel in boolean_mapping:
                new_image[channel, 0] -= 0.33
                new_image[channel, 1] -= 0.33
            concept_image = new_image
            processed_concept_attributions.append(concept_image)

        # Sort concept predictions and corresponding concept attributions and names
        sorted_indices = torch.argsort(concept_predictions, descending=True)  # Sort indices by predictions (high to low)
        concept_predictions = concept_predictions[sorted_indices]
        processed_concept_attributions = [processed_concept_attributions[i] for i in sorted_indices]
        concepts_names = [concepts_names[i] for i in sorted_indices]

        axes[axes_id].imshow(label_attribution_image)
        axes[axes_id].axis('off')
        axes[axes_id].set_title(f'Predicted: {predictClass} | {method_name}')
        axes_id += 1

        # Plot concept attributions
        for idx, concept_image in enumerate(processed_concept_attributions):
            axes[axes_id].imshow(concept_image)
            axes[axes_id].axis('off')
            axes[axes_id].set_title(f'{concepts_names[idx]}: {concept_predictions[idx].item():.2f}')
            axes_id += 1
            if idx == 2:
                break
    plt.savefig(path)

def process_image(model, image, true_label, class_names, attribute_names, device, output_path, args):
    model = model.to(device)
    image = image.to(device)
    
    concept_predictor = ConceptPredictor(model).to(device)
    label_predictor = LabelPredictor(model, args).to(device)

    with torch.no_grad():
        (logits, logits_attri, attributes_logits), _ = model(image)
        
    _, pred_label_idx = torch.max(logits_attri, 1)
    pred_class_name = class_names[pred_label_idx.item()]
    true_class_name = class_names[true_label.item()]
    
    if pred_label_idx.item() != true_label.item():
        print(f"Skipping image - predicted {pred_class_name}, but true label is {true_class_name}")
        return False
    
    top_concept_values, top_concept_indices = torch.topk(attributes_logits.squeeze(), 3)
    
    top_concept_values = torch.sigmoid(top_concept_values)
    
    top_concept_names = [attribute_names[idx.item()] for idx in top_concept_indices]
    
    image.requires_grad_()
    
    label_attr_ig = ig_attribution(label_predictor, image, target_index=pred_label_idx)
    label_attr_ig_smooth = ig_attribution_smooth(label_predictor, image, target_index=pred_label_idx)
    
    concept_attrs_ig = []
    concept_attrs_ig_smooth = []
    
    for idx in top_concept_indices:
        concept_attrs_ig.append(ig_attribution(concept_predictor, image, target_index=idx.item()))
        concept_attrs_ig_smooth.append(ig_attribution_smooth(concept_predictor, image, target_index=idx.item()))
    
    visualize_attribution_new(
        image, 
        [label_attr_ig, label_attr_ig_smooth],
        [concept_attrs_ig, concept_attrs_ig_smooth],
        top_concept_names,
        top_concept_values,
        true_class_name,
        pred_class_name,
        path=output_path
    )
    
    return True

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    model = load_model(args.model_path, device)
    
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    test_dataset = Cub2011AttributeWhole(train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    try:
        classes_name = pd.read_csv(os.path.join(args.data_dir, "CUB_200_2011/classes.txt"), 
                                  sep="\s+", header=None, names=["Index", "Name"])
        class_names = classes_name["Name"].tolist()
    except:
        print("Warning: classes.txt not found. Using generic class names.")
        class_names = [f"Class {i}" for i in range(200)]
        
    try:
        concepts_name = pd.read_csv(os.path.join(args.data_dir, "CUB_200_2011/attributes.txt"), 
                                   sep="\s+", header=None, names=["Index", "Name"])
        attribute_names = concepts_name["Name"].tolist()
    except:
        print("Warning: attributes.txt not found. Using generic attribute names.")
        attribute_names = [f"Attribute {i}" for i in range(112)]
    
    success_count = 0
    for i, (image, label, concepts) in enumerate(test_loader):
        if success_count >= args.num_images:
            break
            
        output_path = os.path.join(args.output_dir, f'attribution_{i}.png')
        success = process_image(model, image, label, class_names, attribute_names, device, output_path, args)
        
        if success:
            success_count += 1
            print(f"Processed image {i}, saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Pixel Attributions for CBM Models')
    parser.add_argument('--model_path', type=str, default='output_cosine/CUB2011/resnet18/1028-1e-4-adam-18-train/checkpoints/epoch-last.pth', help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='datasets', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='attribution_results', help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    parser.add_argument('--num_images', type=int, default=1, help='Number of correctly predicted images to visualize')
    
    args = parser.parse_args()

    main(args)