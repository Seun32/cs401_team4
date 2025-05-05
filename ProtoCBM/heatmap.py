import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

import model
from util.datasets import Cub2011AttributeWhole
from util.preprocess import mean, std
from captum.attr import IntegratedGradients, NoiseTunnel
import pandas as pd

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

class FeaturePredictor(torch.nn.Module):
    def __init__(self, model, feature_type='label'):
        super(FeaturePredictor, self).__init__()
        self.model = model
        self.feature_type = feature_type
        
    def forward(self, x):
        (logits, logits_attri, attributes_logits), _ = self.model(x)
        if self.feature_type == 'label':
            return logits_attri
        elif self.feature_type == 'concept':
            return attributes_logits
        else:
            return logits

def compute_attribution(model, image, target_idx, feature_type='label', use_smoothed=True):
    predictor = FeaturePredictor(model, feature_type=feature_type)
    image.requires_grad_()
    
    ig = IntegratedGradients(predictor)
    
    if use_smoothed:
        smooth_ig = NoiseTunnel(ig)
        attribution = smooth_ig.attribute(
            image, target=target_idx, nt_type='smoothgrad', stdevs=0.1, nt_samples=5)
    else:
        attribution, _ = ig.attribute(image, target=target_idx, return_convergence_delta=True)
    
    return attribution

def visualize_attribution(image, attribution, output_path, threshold=0.01):
    attr_np = attribution.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min())
    
    threshold_value = np.percentile(attr_np, (1 - threshold) * 100)
    mask = attr_np >= threshold_value
    
    original_image = image.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    
    highlighted_image = original_image.copy()
    green_mask = np.zeros_like(original_image)
    green_mask[..., 1] = 1 
    
    alpha = 0.5 
    for c in range(3):
        channel_mask = mask[..., c]
        highlighted_image[channel_mask, 1] = highlighted_image[channel_mask, 1] * (1-alpha) + alpha
    
    plt.figure(figsize=(5, 5))
    plt.imshow(highlighted_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return highlighted_image

def process_image(model, image, label, class_names, device, output_path, args, target_type='label'):
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        (logits, logits_attri, attributes_logits), _ = model(image)
    
    if target_type == 'label':
        _, pred_idx = torch.max(logits_attri, 1)
        target_idx = pred_idx.item()
    elif target_type == 'concept':
        target_idx = torch.argmax(attributes_logits).item()
    else:
        target_idx = label.item() 
    
    attribution = compute_attribution(model, image, target_idx, feature_type=target_type, 
                                     use_smoothed=True)
    
    highlighted = visualize_attribution(image, attribution, output_path, threshold=args.threshold)
    
    return highlighted

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
    
    for i, (image, label, _) in enumerate(test_loader):
        if i >= args.num_images:
            break
            
        output_path = os.path.join(args.output_dir, f'highlight_{i}.png')
        
        highlighted = process_image(model, image, label, class_names, device, 
                                   output_path, args, target_type=args.target_type)
        
        print(f"Processed image {i}, saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Simple Pixel Attributions')
    parser.add_argument('--model_path', type=str, default='output_cosine/CUB2011/resnet18/1028-1e-4-adam-18-train/checkpoints/epoch-last.pth', help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='datasets', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='highlight_results', help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to process')
    parser.add_argument('--threshold', type=float, default=0.01, help='Threshold for attribution visualization (lower = more highlighted areas)')
    parser.add_argument('--target_type', type=str, default='label', choices=['label', 'concept', 'true_label'], help='Type of target to generate attribution for')
    
    args = parser.parse_args()
    main(args)