import os, torch, sys, joblib
#os.environ["CUDA_VISIBLE_DEVICES"]="6"
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.cm as cm
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scipy.ndimage import sobel
from dataset import load_data
from scipy.ndimage import gaussian_filter
from config import BASE_DIR, N_CLASSES, N_ATTRIBUTES
from captum.attr import IntegratedGradients, ShapleyValueSampling, Saliency, Lime, NoiseTunnel
from matplotlib.patches import Rectangle
import torch.serialization
from CUB.template_model import End2EndModel
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
    

def visualize_attribution_new(inputs, label_attributions_list, concept_attributions_list, concepts_names, concept_predictions, className, predictClass, path=None):
    original_image = inputs.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    axes_id = 0
    total_images = 1 + len(label_attributions_list) * 4
    fig, axes = plt.subplots(1, total_images, figsize=(10 + 5 * 16, 5))

    # Plot original image
    axes[axes_id].imshow(original_image)
    axes[axes_id].axis('off')
    axes[axes_id].set_title(f'{className}')
    axes_id += 1

    label_threshold = 0.05
    concept_threshold = 0.001
    alpha = 0.6  # overlay strength

    for label_attributions, concept_attributions, method_name in zip(
        label_attributions_list,
        concept_attributions_list,
        ["IG", "smoothed IG", "saliency", "smoothed saliency"]
    ):
        # Label Attribution Overlay
        label_attr = label_attributions.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
        label_attr = (label_attr - label_attr.min()) / (label_attr.max() - label_attr.min())
        label_thresh_val = np.percentile(label_attr, (1 - label_threshold) * 100)
        mask = np.where(label_attr >= label_thresh_val, 1.0, 0.0)
        if mask.shape[-1] > 1:
            mask = np.mean(mask, axis=-1)
        green_overlay = np.zeros_like(original_image)
        green_overlay[..., 1] = mask
        overlayed_label = np.clip((1 - alpha) * original_image + alpha * green_overlay, 0, 1)

        axes[axes_id].imshow(overlayed_label)
        axes[axes_id].axis('off')
        axes[axes_id].set_title(f'Predicted: {predictClass} | {method_name}')
        axes_id += 1

        # Concept Attribution Overlays
        processed_concept_overlays = []
        for concept_attribution in concept_attributions:
            concept_attr = concept_attribution.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
            concept_attr = (concept_attr - concept_attr.min()) / (concept_attr.max() - concept_attr.min())
            concept_thresh_val = np.percentile(concept_attr, (1 - concept_threshold) * 100)
            mask = np.where(concept_attr >= concept_thresh_val, 1.0, 0.0)
            if mask.shape[-1] > 1:
                mask = np.mean(mask, axis=-1)
            green_overlay = np.zeros_like(original_image)
            green_overlay[..., 1] = mask
            overlayed_concept = np.clip((1 - alpha) * original_image + alpha * green_overlay, 0, 1)
            processed_concept_overlays.append(overlayed_concept)

        # Sort by top predicted concepts
        sorted_indices = torch.argsort(concept_predictions, descending=True)
        concept_predictions = concept_predictions[sorted_indices]
        processed_concept_overlays = [processed_concept_overlays[i] for i in sorted_indices]
        concepts_names = [concepts_names[i] for i in sorted_indices]

        for idx, overlayed_concept in enumerate(processed_concept_overlays):
            axes[axes_id].imshow(overlayed_concept)
            axes[axes_id].axis('off')
            axes[axes_id].set_title(f'{concepts_names[idx]}: {concept_predictions[idx].item():.2f}')
            axes_id += 1
            if idx == 2:
                break

    plt.tight_layout()
    if path:
        plt.savefig(path)
    else:
        plt.show()



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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    classes_name = pd.read_csv("./classes.txt", sep="\s+", header=None, names=["Index", "Name"])
    concepts_name = pd.read_csv("./attributes.txt", sep="\s+", header=None, names=["Index", "Name"])
    if args.model == "joint":
        # model = torch.load(args.model_dirs)
        torch.serialization.add_safe_globals([End2EndModel])
        # model = torch.load(args.model_dirs, weights_only=False)
        model = torch.load(args.model_dirs, map_location=torch.device('cpu'), weights_only=False)

        model.eval()
        labelPredictor = LabelPredictor(model.first_model, model.sec_model, args).to(device)
        conceptPredictor = ConceptPredictor(model.first_model).to(device)

    elif args.model == "independent":
        model1 = torch.load(args.model_dirs, weights_only=False).to(device)
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

        model2 = torch.load(args.model_dirs2, weights_only=False).to(device)
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

    for data_idx, data in enumerate(loader):
        inputs, labels, attr_labels, path = data

        # UNCOMMENT the below code and modify it appropriately if you seek to process certain bird files/folders
        # if "White_Pelican" not in path[0]:
        #     continue

        attr_labels = torch.stack(attr_labels).t()
        attr_labels = attr_labels[:, :112] # Only keep the first 112 attributes

        inputs_var = torch.autograd.Variable(inputs).to(device)
        labels_var = torch.autograd.Variable(labels).to(device)

        concepts_outputs = conceptPredictor(inputs_var)

        className = classes_name["Name"][labels.squeeze().item()]

        label_outputs = labelPredictor(inputs_var)
        prediction_score, pred_label_idx = torch.topk(label_outputs, 1)

        filename = os.path.basename(path[0])

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

            concepts_predictions = (concepts_predictions - concepts_predictions.min()) / (concepts_predictions.max() - concepts_predictions.min()) #normalize

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
# python3 CUB/feature_attribution.py -model joint 
    # -model_dirs /home/konghaoz/cbm/CUB/outputs/jointModel/best_model_1.pt -outputDir /home/konghaoz/cbm/CUB/outputs/jointModel/ensemble -use_attr
# python3 CUB/feature_attribution.py -model independent -model_dirs CUB/outputs/conceptModel/best_model_1.pt -model_dirs2 CUB/outputs/labelModel/best_model_1.pt -outputDir /home/konghaoz/cbm/CUB/outputs/conceptModel/ensemble -use_sigmoid -use_attr