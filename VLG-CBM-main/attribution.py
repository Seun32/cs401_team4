import os, torch, sys
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import matplotlib.pyplot as plt
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from captum.attr import IntegratedGradients, Saliency, NoiseTunnel
from model.cbm import CBM_model
from data import utils as data_utils

SMOOTH_SAMPLES = 5
SMOOTH_STDEV = 0.1

class LabelPredictor(torch.nn.Module):
    def __init__(self, model):
        super(LabelPredictor, self).__init__()
        self.model = model

    def forward(self, x):
        outputs, _ = self.model(x)
        return outputs

class ConceptPredictor(torch.nn.Module):
    def __init__(self, model):
        super(ConceptPredictor, self).__init__()
        self.model = model

    def forward(self, x):
        _, concepts = self.model(x)
        return concepts

def process(attribution, normalize=True):
    """Process attribution map for visualization"""
    attr_np = attribution.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    if normalize:
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
    return attr_np

def visualize(inputs, label_attributions_list, concept_attributions_list, 
                            concepts_names, concept_predictions, className, predictClass, path=None):
    original_image = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    num_methods = len(label_attributions_list)
    fig, axes = plt.subplots(num_methods, 4, figsize=(25, 5 * num_methods))
    
    if num_methods == 1:
        axes = axes.reshape(1, -1)

    label_threshold = 0.05 
    concept_threshold = 0.001 

    method_names = ["IG", "Smoothed IG", "Saliency", "Smoothed Saliency"]
    for method_idx, (label_attributions, concept_attributions, method_name) in enumerate(
            zip(label_attributions_list, concept_attributions_list, method_names)):
        
        label_attr = process(label_attributions)
        label_threshold_value = np.percentile(label_attr, (1 - label_threshold) * 100)
        label_mask = (label_attr >= label_threshold_value).any(axis=-1)

        processed_concepts = []
        for concept_attr in concept_attributions:
            concept_map = process(concept_attr)
            thresh_value = np.percentile(concept_map, (1 - concept_threshold) * 100)
            concept_mask = (concept_map >= thresh_value).any(axis=-1)
            processed_concepts.append(concept_mask)

        sorted_indices = torch.argsort(concept_predictions, descending=True)
        concept_predictions_sorted = concept_predictions[sorted_indices]
        processed_concepts = [processed_concepts[i] for i in sorted_indices]
        concepts_names_sorted = [concepts_names[i] for i in sorted_indices]

        ax = axes[method_idx, 0]
        ax.imshow(original_image)
        overlay = np.zeros((*label_mask.shape[:2], 4))
        overlay[label_mask, :] = [1, 0, 0, 0.4]
        ax.imshow(overlay)
        ax.axis('off')
        ax.set_title(f'{method_name}\nPred: {predictClass}\nTrue: {className}', fontsize=10)

        for i in range(3):
            ax = axes[method_idx, i + 1]
            ax.imshow(original_image)
            concept_mask = processed_concepts[i]
            overlay = np.zeros((*concept_mask.shape[:2], 4))
            overlay[concept_mask, :] = [0, 1, 0, 0.8]
            ax.imshow(overlay)
            ax.axis('off')
            ax.set_title(f'{concepts_names_sorted[i]}\n({concept_predictions_sorted[i]:.2f})', 
                        fontsize=8)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def compute(model, inputs, target_idx, method='ig', smooth=False):
    """Unified attribution computation function"""
    inputs.requires_grad_()
    
    if method == 'ig':
        ig = IntegratedGradients(model)
        if smooth:
            noise_tunnel = NoiseTunnel(ig)
            return noise_tunnel.attribute(
                inputs, 
                target=target_idx,
                nt_type='smoothgrad',
                nt_samples=SMOOTH_SAMPLES,
                stdevs=SMOOTH_STDEV
            )
        else:
            return ig.attribute(inputs, target=target_idx)[0]
    
    elif method == 'saliency':
        saliency = Saliency(model)
        if smooth:
            noise_tunnel = NoiseTunnel(saliency)
            return noise_tunnel.attribute(
                inputs,
                target=target_idx,
                nt_type='smoothgrad',
                nt_samples=SMOOTH_SAMPLES,
                stdevs=SMOOTH_STDEV
            )
        else:
            return saliency.attribute(inputs, target=target_idx)

def main(args):
    if args.model == 'joint':
        model = CBM_model.load_from_checkpoint(args.model_dirs)
    elif args.model == 'independent':
        model = CBM_model.load_from_checkpoint(args.model_dirs)
        if args.use_sigmoid:
            model.sigmoid = True
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(args.device)
    model.eval()

    label_predictor = LabelPredictor(model)
    concept_predictor = ConceptPredictor(model)

    concepts = data_utils.get_concepts(args.dataset)
    classes = data_utils.get_classes(args.dataset)

    os.makedirs(args.output_dir, exist_ok=True)

    test_loader = data_utils.get_test_loader(
        args.dataset,
        batch_size=1,
        shuffle=False
    )

    for data_idx, (inputs, labels, concept_labels) in enumerate(test_loader):
        inputs = inputs.to(args.device)
        
        with torch.no_grad():
            outputs, concept_outputs = model(inputs)
            _, pred_label_idx = torch.max(outputs, 1)

        if labels.item() == pred_label_idx.item():
            attributions = {
                'ig': compute(label_predictor, inputs, pred_label_idx, 'ig', False),
                'ig_smooth': compute(label_predictor, inputs, pred_label_idx, 'ig', True),
                'saliency': compute(label_predictor, inputs, pred_label_idx, 'saliency', False),
                'saliency_smooth': compute(label_predictor, inputs, pred_label_idx, 'saliency', True)
            }

            concept_indices = torch.nonzero(concept_labels == 1, as_tuple=True)[1]
            concept_predictions = concept_outputs[0, concept_indices]
            concept_names = [concepts[i] for i in concept_indices]

            concept_attributions = {method: [] for method in attributions.keys()}
            for idx in concept_indices:
                for method in attributions.keys():
                    is_smooth = 'smooth' in method
                    base_method = 'ig' if 'ig' in method else 'saliency'
                    attr = compute(concept_predictor, inputs, idx, base_method, is_smooth)
                    concept_attributions[method].append(attr)

            label_attr_list = [
                attributions['ig'],
                attributions['ig_smooth'],
                attributions['saliency'],
                attributions['saliency_smooth']
            ]
            
            concept_attr_list = [
                concept_attributions['ig'],
                concept_attributions['ig_smooth'],
                concept_attributions['saliency'],
                concept_attributions['saliency_smooth']
            ]

            visualize(
                inputs[0],
                label_attr_list,
                concept_attr_list,
                concept_names,
                concept_predictions,
                classes[labels.item()],
                classes[pred_label_idx.item()],
                path=os.path.join(args.output_dir, f"attribution_{data_idx}.png")
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Attribution for CBM')
    parser.add_argument('-model', type=str, required=True,
                        choices=['joint', 'independent'],
                        help='Model type (joint or independent)')
    parser.add_argument('-model_dirs', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('-model_dirs2', type=str,
                        help='Path to second model checkpoint (for independent model)')
    parser.add_argument('-outputDir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('-use_sigmoid', action='store_true',
                        help='Use sigmoid activation')
    parser.add_argument('-use_attr', action='store_true',
                        help='Generate attributions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    
    args = parser.parse_args()
    main(args)
    