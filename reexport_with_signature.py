# reexport_with_signature.py
# Monkey-patch absl.flags.DEFINE_integer to ignore duplicate flag definitions
import os
from absl import flags
from absl.flags import _exceptions

# Backup original DEFINE_integer
_orig_define_integer = flags.DEFINE_integer

def _safe_define_integer(name, default, doc, **kwargs):
    try:
        return _orig_define_integer(name, default, doc, **kwargs)
    except _exceptions.DuplicateFlagError:
        return flags.FLAGS[name]

# Apply monkey-patch
flags.DEFINE_integer = _safe_define_integer

# Import before model defines flags
import tensorflow as tf
from concept_bottleneck_model import ConceptBottleneckModel
from tf_dataset import get_dataset, num_classes
import pandas as pd

# Restore DEFINE_integer
flags.DEFINE_integer = _orig_define_integer

# Override intermediate_size flag to match training
flags.FLAGS['intermediate_size'].value = 312

# Define script flags
parser = flags.FLAGS
flags.DEFINE_string('saved_model_dir', None,
                    'Directory of original SavedModel (saved_model.pb + variables/)')
flags.DEFINE_string('attributes_file', 'attributes.txt',
                    'Path to attribute names file (one per line)')
flags.DEFINE_string('export_dir', 'signed_model',
                    'Output directory for re-exported model')

if __name__ == '__main__':
    import sys
    # Parse flags via absl
    flags.FLAGS(sys.argv)

    # Load attribute names to get concept count
    concept_df = pd.read_csv(
        flags.FLAGS.attributes_file,
        sep="\s+", header=None, names=["Idx","Name"]
    )
    N_CONCEPTS = len(concept_df)
    print(f"Using {N_CONCEPTS} concepts from {flags.FLAGS.attributes_file}")

    # 1) Reconstruct the CBM architecture
    LATENT_DIMS = 0
    N_CLASSES   = num_classes
    MODEL_TYPE  = 'joint'

    model = ConceptBottleneckModel(
        N_CONCEPTS,
        LATENT_DIMS,
        N_CLASSES,
        use_inceptionv3=True,
        type=MODEL_TYPE
    )

    # 2) Restore weights
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_prefix = os.path.join(
        flags.FLAGS.saved_model_dir, 'variables', 'variables'
    )
    ckpt.restore(ckpt_prefix).expect_partial()
    print(f"Restored weights from {ckpt_prefix}*")

    # 3) Define a serving function with correct shapes
    @tf.function(input_signature=[
        tf.TensorSpec([None, 299, 299, 3], tf.float32, name='inputs')
    ])
    def serve(inputs):
        # concept logits via the model's concept head
        # assume model.intermediate_predictor outputs N_CONCEPTS logits
        c_logits = model.intermediate_predictor(inputs, training=False)
        # apply sigmoid if your model used it
        c_probs  = tf.nn.sigmoid(c_logits)
        # class logits via the model's label head
        y_logits = model.label_predictor(c_probs, training=False)
        return {
            'concept_predictions': c_logits,
            'class_predictions':   y_logits,
        }

    # 4) Export with explicit signature
    tf.saved_model.save(
        model,
        export_dir=flags.FLAGS.export_dir,
        signatures={'serving_default': serve}
    )
    print(f"Re-exported model with signature → {flags.FLAGS.export_dir}/")
