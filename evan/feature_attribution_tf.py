import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import textwrap

# parse absl flags first so tf_dataset can see them
from absl import flags
from absl.flags import _exceptions
try:
    flags.FLAGS(sys.argv)
except _exceptions.UnrecognizedFlagError:
    flags.FLAGS([''])

from tf_dataset import get_dataset, num_classes

parser = argparse.ArgumentParser(description="TF Feature Attribution (speeded‑up)")
parser.add_argument("--saved_model_dir", required=True,
                    help="folder containing saved_model.pb and variables/")
parser.add_argument("--output_dir", required=True,
                    help="where to dump attribution images")
parser.add_argument("--eval_set", default="test", choices=["train","test","all"],
                    help="which split to run on")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch size for attribution")
args = parser.parse_args()

# --- HYPERPARAMS FOR SPEED ---
IG_STEPS       = 25
SMOOTH_SAMPLES =  5
SMOOTH_STDEV   = 0.1
TOP_K_PIXELS   = 500   # number of top pixels to highlight

# -----------------------------------------------------------------------------
def normalize_img(x):
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

@tf.function
def batched_ig(fd_infer, inp, target, output_key="class_predictions", steps=IG_STEPS):
    inp = tf.cast(inp, tf.float32)
    baseline = tf.zeros_like(inp)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps+1), [steps+1,1,1,1])
    scaled = baseline + alphas * (inp - baseline)

    with tf.GradientTape() as tape:
        tape.watch(scaled)
        out = fd_infer(inputs=scaled)
        logits = out[output_key][:, target]
    grads = tape.gradient(logits, scaled)
    avg_grad = tf.reduce_mean(grads[:-1], axis=0)
    delta = inp[0] - baseline[0]
    return delta * avg_grad


def compute_ig(infer, inp, target):
    return batched_ig(infer, inp, target)


def compute_ig_smooth(infer, inp, target):
    maps = []
    for _ in range(SMOOTH_SAMPLES):
        noise = tf.random.normal(shape=inp.shape, stddev=SMOOTH_STDEV)
        maps.append(compute_ig(infer, inp + noise, target))
    return tf.reduce_mean(tf.stack(maps,0), axis=0).numpy()

# -----------------------------------------------------------------------------
def visualize_all(inp,
                  ig_map, ig_smooth_map,
                  concept_ig, concept_ig_smooth,
                  concept_names, concept_vals,
                  class_name, predict_class,
                  outpath):
    img = inp.numpy().squeeze()
    img_norm = normalize_img(img)

    fig, axes = plt.subplots(1, 11, figsize=(33, 3))
    fig.subplots_adjust(wspace=0.7)
    for ax in axes:
        ax.axis('off')

    # Original image
    axes[0].imshow(img_norm)
    title0 = f"True: {class_name}\nPred: {predict_class}"
    axes[0].set_title('\n'.join(textwrap.wrap(title0, 20)), fontsize=10)

    def plot_topk_mask(ax, heat, title):
        h = normalize_img(heat)
        # collapse channels
        if h.ndim == 3:
            h = h.mean(axis=2)
        # find threshold for top K pixels
        flat = h.flatten()
        if TOP_K_PIXELS < flat.size:
            thresh = np.partition(flat, -TOP_K_PIXELS)[-TOP_K_PIXELS]
        else:
            thresh = flat.min()
        mask = (h >= thresh).astype(float)
        highlighted = mask
        overlay = np.zeros_like(img_norm)
        overlay[..., 1] = highlighted
        ax.imshow(img_norm, cmap=None)
        # make a 3‑channel “mask image” which is zero everywhere except green=1 at mask
        overlay = np.zeros_like(img_norm)
        overlay[...,1] = mask  
        # now blend *only* the mask region
        ax.imshow(overlay, alpha=0.7) 
        wrapped = '\n'.join(textwrap.wrap(title, 20))
        ax.set_title(wrapped, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # IG: class and heat
    plot_topk_mask(axes[1], ig_map, "IG: class")
    plot_topk_mask(axes[2], ig_map, "IG: heat")

    # IG: top-3 concept attributions
    for i in range(3):
        plot_topk_mask(
            axes[3+i],
            concept_ig[i],
            f"{concept_names[i]} ({concept_vals[i]:.2f})"
        )

    # Smoothed IG: class and heat
    plot_topk_mask(axes[6], ig_smooth_map, "Smoothed IG")
    plot_topk_mask(axes[7], ig_smooth_map, "sIG: heat")

    # Smoothed IG: top-3 concepts
    for i in range(3):
        plot_topk_mask(
            axes[8+i],
            concept_ig_smooth[i],
            f"{concept_names[i]} (sIG)"
        )

    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    module = tf.saved_model.load(args.saved_model_dir)
    raw_infer = module.signatures['serving_default']
    _, kw_specs = raw_infer.structured_input_signature

    inputs_spec = kw_specs['inputs']
    @tf.function(input_signature=[inputs_spec])
    def infer(inputs): return raw_infer(inputs=inputs)

    count = 0
    datasets = []
    if args.eval_set in ("all","train"):
        datasets.append(("train", get_dataset(True, False).batch(args.batch_size)))
    if args.eval_set in ("all","test"):
        datasets.append(("test",  get_dataset(False,False).batch(args.batch_size)))

    for split, ds in datasets:
        for batch in ds:
            x = batch['data']
            y_true = batch['label']
            out = infer(inputs=x)
            c_out = out['concept_predictions'].numpy()
            y_out = out['class_predictions']
            y_pred = tf.argmax(y_out, axis=1, output_type=y_true.dtype)

            for i in tf.where(y_true==y_pred)[:,0].numpy():
                inp = x[i:i+1]
                t = int(y_pred[i])

                ig_map = compute_ig(infer, inp, t).numpy()
                ig_smooth_map = compute_ig_smooth(infer, inp, t)

                top_idx = np.argsort(-c_out[i])[:3]
                cnames = [concept_names[j] for j in top_idx]
                cvals = c_out[i, top_idx]
                c_ig = [batched_ig(infer, inp, j, output_key='concept_predictions').numpy()
                       for j in top_idx]
                c_ig_smooth = [np.mean([batched_ig(
                                            infer,
                                            inp + tf.random.normal(inp.shape,SMOOTH_STDEV),
                                            j,
                                            output_key='concept_predictions'
                                        ).numpy() for _ in range(SMOOTH_SAMPLES)],0)
                               for j in top_idx]

                outpath = os.path.join(args.output_dir, f"{split}_{count:03d}.png")
                visualize_all(
                    x[i], ig_map, ig_smooth_map,
                    c_ig, c_ig_smooth,
                    cnames, cvals,
                    classes_name[y_true[i].numpy()],
                    classes_name[t],
                    outpath
                )
                count += 1
                if count>=10:
                    return

if __name__ == '__main__':
    classes_name = pd.read_csv(
        'CUB_200_2011/classes.txt', sep=r"\s+",
        header=None, names=['Idx','Name'])['Name'].tolist()
    concept_names = pd.read_csv(
        'attributes.txt', sep=r"\s+",
        header=None, names=['Idx','Name'])['Name'].tolist()
    main()
