import os, torch, sys, joblib
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scipy.ndimage import sobel
from CUB.dataset import load_data
from scipy.ndimage import gaussian_filter
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from captum.attr import IntegratedGradients, ShapleyValueSampling, Saliency, Lime, NoiseTunnel
from matplotlib.patches import Rectangle
# from captum.attr import LayerConductance
# from captum.attr import NeuronConductance

class LabelPredictor(torch.nn.Module):
    def __init__(self, model1, model2, args):
        super(LabelPredictor, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.args = args

    def forward(self, x):
        outputs = self.first_model(x)
        if self.args.use_sigmoid:
            attr_outputs = torch.cat([torch.nn.Sigmoid()(o) for o in outputs], dim=1)
        else:
            attr_outputs = torch.cat(outputs, dim=1)
        
        return self.sec_model(attr_outputs)

class ConceptPredictor(torch.nn.Module):
    def __init__(self, model1):
        super(ConceptPredictor, self).__init__()
        self.conceptModel = model1

    def forward(self, x):
        return torch.cat(self.conceptModel(x), dim=1)

# ... (previous imports and classes remain the same)

def visualize_attribution_new(inputs, label_attributions_list, concept_attributions_list, concepts_names, concept_predictions, className, predictClass, path=None):
    original_image = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # Total subplots: 1 row per method (original + overlay + 3 concepts)
    num_methods = len(label_attributions_list)
    total_images = num_methods * (1 + 3)  # 1 overlay + 3 concepts per method
    fig, axes = plt.subplots(num_methods, 4, figsize=(25, 5 * num_methods))  # Adjust layout
    
    if num_methods == 1:
        axes = axes.reshape(1, -1)  # Ensure axes is 2D for consistency

    label_threshold = 0.05
    concept_threshold = 0.001

    for method_idx, (label_attributions, concept_attributions, method_name) in enumerate(zip(label_attributions_list, concept_attributions_list, ["IG", "smoothed IG", "saliency", "smoothed saliency"])):
        # Process label attribution
        label_attribution_image = label_attributions.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
        label_attribution_image = (label_attribution_image - label_attribution_image.min()) / (label_attribution_image.max() - label_attribution_image.min())
        label_threshold_value = np.percentile(label_attribution_image, (1 - label_threshold) * 100)
        label_mask = (label_attribution_image >= label_threshold_value).any(axis=-1)

        # Process concept attributions
        processed_concept_attributions = []
        for concept_attribution in concept_attributions:
            concept_image = concept_attribution.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
            concept_image = (concept_image - concept_image.min()) / (concept_image.max() - concept_image.min())
            concept_threshold_value = np.percentile(concept_image, (1 - concept_threshold) * 100)
            concept_mask = (concept_image >= concept_threshold_value).any(axis=-1)
            processed_concept_attributions.append(concept_mask)

        # Sort concepts by prediction confidence
        sorted_indices = torch.argsort(concept_predictions, descending=True)
        concept_predictions_sorted = concept_predictions[sorted_indices]
        processed_concept_attributions = [processed_concept_attributions[i] for i in sorted_indices]
        concepts_names_sorted = [concepts_names[i] for i in sorted_indices]

        # --- Plotting Changes Start Here ---
        # Plot original + label attribution overlay
        ax = axes[method_idx, 0] if num_methods > 1 else axes[0]
        ax.imshow(original_image)
        # Overlay label attribution (red mask)
        overlay = np.zeros((*label_mask.shape[:2], 4))  # RGBA array
        overlay[label_mask, :] = [1, 0, 0, 0.4]  # Red with 40% opacity
        ax.imshow(overlay)
        ax.axis('off')
        ax.set_title(f'Predicted: {predictClass} | {method_name}')

        # Plot top 3 concept attributions
        for i in range(3):
            ax = axes[method_idx, i + 1] if num_methods > 1 else axes[i + 1]
            ax.imshow(original_image)
            # Overlay concept attribution (yellow mask)
            concept_mask = processed_concept_attributions[i]
            overlay = np.zeros((*concept_mask.shape[:2], 4))
            overlay[concept_mask, :] = [1, 1, 0, 0.4]  # Yellow with 40% opacity
            ax.imshow(overlay)
            ax.axis('off')
            ax.set_title(f'{concepts_names_sorted[i]}: {concept_predictions_sorted[i].item():.2f}')

    plt.tight_layout()
    plt.savefig(path)

def ig_attribution(model, inputs, target_index, return_convergence_delta=True):
    inputs.requires_grad_()
    ig = IntegratedGradients(model)
    attribution, delta = ig.attribute(inputs, target = target_index, return_convergence_delta=return_convergence_delta)
    return attribution

def shap_attribution(model, inputs, target_index, n_samples=50):
    shap = ShapleyValueSampling(model)
    attribution = shap.attribute(inputs, target=target_index, n_samples=n_samples)
    return attribution

def lime_attribution(model, inputs, target_index, n_samples=50):
    lime = Lime(model)
    attribution = lime.attribute(inputs, target=target_index, n_samples=n_samples)
    return attribution

def saliency_attribution(model, inputs, target_index):
    inputs.requires_grad_()
    saliency = Saliency(model)
    attribution = saliency.attribute(inputs, target=target_index)
    return attribution

def ig_attribution_smooth(model, inputs, target_index):
    inputs.requires_grad_()
    ig = IntegratedGradients(model)
    smooth_ig = NoiseTunnel(ig)
    attribution = smooth_ig.attribute(
        inputs, target=target_index, nt_type='smoothgrad', stdevs=0.1, nt_samples=5)
    return attribution

def saliency_attribution_smooth(model, inputs, target_index):
    inputs.requires_grad_()
    saliency = Saliency(model)
    smoothgrad = NoiseTunnel(saliency)
    attribution = smoothgrad.attribute(inputs, target=target_index, nt_type='smoothgrad', stdevs=0.01, nt_samples=5)
    return attribution

def main(args):
    classes_name = pd.read_csv("../CUB/CUB_200_2011/CUB_200_2011/classes.txt", sep=r"\s+", header=None, names=["Index", "Name"])
    concepts_name = pd.read_csv("../CUB/CUB_200_2011/attributes.txt", sep=r"\s+", header=None, names=["Index", "Name"])

    if args.model == "joint":
        model = torch.load(args.model_dirs)
        model.eval()
        labelPredictor = LabelPredictor(model.first_model, model.sec_model).cuda()   
        conceptPredictor = ConceptPredictor(model.first_model).cuda()
   
    elif args.model == "independent":
        model1 = torch.load(args.model_dirs, weights_only=False).cuda()  
        if not hasattr(model1, 'use_relu'):
            if args.use_relu:
                model1.use_relu = True
            else:
                model1.use_relu = False
        if not hasattr(model1, 'use_sigmoid'):
            if args.use_sigmoid:
                model1.use_sigmoid = True
            else:
                model1.use_sigmoid = False
        if not hasattr(model1, 'cy_fc'):
            model1.cy_fc = None
        model1.eval()

        model2 = torch.load(args.model_dirs2, weights_only=False).cuda()  
        if not hasattr(model2, 'use_relu'):
            if args.use_relu:
                model2.use_relu = True
            else:
                model2.use_relu = False
        if not hasattr(model2, 'use_sigmoid'):
            if args.use_sigmoid:
                model2.use_sigmoid = True
            else:
                model2.use_sigmoid = False
        model2.eval()

        conceptPredictor = ConceptPredictor(model1)
        labelPredictor = LabelPredictor(model1, model2, args)

    data_dir = os.path.join(BASE_DIR, args.data_dir, args.eval_data + '.pkl')
    loader = load_data([data_dir], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir,
                       n_class_attr=args.n_class_attr)

    count = 0
    for data_idx, data in enumerate(loader):
        count += 1
        
        inputs, labels, attr_labels, path = data
        attr_labels = torch.stack(attr_labels).t() 
        attr_labels = attr_labels[:, :112]  # Only keep the first 112 attributes


        inputs_var = torch.autograd.Variable(inputs).cuda() 
        labels_var = torch.autograd.Variable(labels).cuda() 

        concepts_outputs = conceptPredictor(inputs_var)

        className = classes_name["Name"][labels.squeeze().item()]

        label_outputs = labelPredictor(inputs_var)
        prediction_score, pred_label_idx = torch.topk(label_outputs, 1)
    
        if labels.squeeze().item() == pred_label_idx.squeeze().item():
            # label attribution
            predictClass = classes_name["Name"][pred_label_idx.squeeze().item()]
            labelAttr_saliency = saliency_attribution(labelPredictor, inputs_var, target_index=pred_label_idx)[0]
            labelAttr_saliency_smooth = saliency_attribution_smooth(labelPredictor, inputs_var, target_index=pred_label_idx)[0]
            labelAttr_ig = ig_attribution(labelPredictor, inputs_var, target_index=pred_label_idx)[0]
            labelAttr_ig_smooth = ig_attribution_smooth(labelPredictor, inputs_var, target_index=pred_label_idx)[0]
            # concepts attribution
            concepts_indices_list = torch.nonzero(attr_labels == 1, as_tuple=True)[1].tolist()
            concepts_indices_list = [i for i in concepts_indices_list if i < 112]

            if len(concepts_indices_list) == 0:
                print("No active attributes found for this sample.")
                # You can choose to skip visualization for this sample
                continue
            concepts_names = concepts_name["Name"][concepts_indices_list].tolist()
            print("concepts_indices_list:", concepts_indices_list)

            concepts_predictions = concepts_outputs[0][concepts_indices_list]

            concepts_predictions = (concepts_predictions - concepts_predictions.min()) / (concepts_predictions.max()  - concepts_predictions.min()) #normalize
            
            concepts_attributions_ig = []
            concepts_attributions_ig_smooth = []
            concepts_attributions_saliency = []
            concepts_attributions_saliency_smooth = []
            for index in concepts_indices_list:
                concepts_attributions_ig.append(ig_attribution(conceptPredictor, inputs_var, target_index=torch.tensor([[index]]))[0])
                concepts_attributions_ig_smooth.append(ig_attribution_smooth(conceptPredictor, inputs_var, target_index=torch.tensor([[index]]))[0])
                concepts_attributions_saliency.append(saliency_attribution(conceptPredictor, inputs_var, target_index=torch.tensor([[index]]))[0])
                concepts_attributions_saliency_smooth.append(saliency_attribution_smooth(conceptPredictor, inputs_var, target_index=torch.tensor([[index]]))[0])

            visualize_attribution_new(inputs_var[0], (labelAttr_ig, labelAttr_ig_smooth, labelAttr_saliency, labelAttr_saliency_smooth), (concepts_attributions_ig, concepts_attributions_ig_smooth, concepts_attributions_saliency, concepts_attributions_saliency_smooth), concepts_names, concepts_predictions, className, predictClass,
                            path=f"{os.path.join(args.outputDir, path[0].split('/')[-1])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concepts Validation')
    parser.add_argument('-model_dirs', default=None, help='where the trained models are saved')
    parser.add_argument('-model_dirs2', default=None, help='where the trained models are saved')
    parser.add_argument('-eval_data', default='test', help='Type of data (train/ val/ test) to be used')
    parser.add_argument('-use_attr', default = True, help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)', action='store_true')
    parser.add_argument('-no_img', help='if included, only use attributes (and not raw imgs) for class prediction', action='store_true')
    parser.add_argument('-image_dir', default='images', help='test image folder to run inference on')
    parser.add_argument('-n_class_attr', type=int, default=2, help='whether attr prediction is a binary or triary classification')
    parser.add_argument('-data_dir', default='CUB_processed/class_attr_data_10', help='directory to the data used for evaluation')
    parser.add_argument('-n_attributes', type=int, default=N_ATTRIBUTES, help='whether to apply bottlenecks to only a few attributes')    
    parser.add_argument('-model', default=None, help='whether the model is a joint cbm or independent cbm')    
    parser.add_argument('-outputDir', default=None, help='output directory')  
    parser.add_argument('-use_sigmoid', help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')
    parser.add_argument('-use_relu', help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model', action='store_true')

 
    args = parser.parse_args()
    args.batch_size = 1

    main(args)
    # python3 CUB/feature_attribution.py -model joint -model_dirs /home/konghaoz/cbm/CUB/outputs/jointModel/best_model_1.pt -outputDir /home/konghaoz/cbm/CUB/outputs/jointModel/ensemble -use_attr
    # python3 CUB/feature_attribution.py -model independent -model_dirs CUB/outputs/conceptModel/best_model_1.pt -model_dirs2 CUB/outputs/labelModel/best_model_1.pt -outputDir /home/konghaoz/cbm/CUB/outputs/conceptModel/ensemble -use_sigmoid -use_attr

