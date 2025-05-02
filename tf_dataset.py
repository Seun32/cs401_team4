import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pathlib
import os
from absl import app
from absl import flags
import csv

FLAGS = flags.FLAGS

flags.DEFINE_string('class_concepts_path', './attributes_and_labels/class_concepts.npy', 'Npy file containing the concepts for each label.')

image_size = 299
num_classes = 200

def get_concept_groups():
  concept_names = []
  with open('./attributes_and_labels/concept_names.txt', 'r') as f:
    for line in f:
      concept_names.append(line.replace('\n', '').split('::'))

  group_names = []
  for c in concept_names:
    if c[0] not in group_names:
      group_names.append(c[0])
  groups = np.zeros((len(group_names), len(concept_names)), dtype=np.float32)
  for i, gn in enumerate(group_names):
    for j, cn in enumerate(concept_names):
      if cn[0] == gn:
        groups[i, j] = 1.
  return groups

def get_dataset(train, augmented):
    """Creates a tf.data.Dataset directly from local CUB_200_2011 folder."""

    # Load the concept vectors, shape [200, num_concepts]
    class_concepts = tf.constant(np.load(FLAGS.class_concepts_path))

    def parse_image(path):
        """
        1) Read and decode the image file from disk.
        2) Derive a numeric label from the folder name, e.g. "001.Black_footed_Albatross" -> label=0
        3) Return a dict with 'image' and 'label'.
        """
        # Read file
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image_bytes, channels=3)

        # Extract folder name: "001.Black_footed_Albatross"
        folder_name = tf.strings.split(path, os.sep)[-2]
        # Split at '.' to get "001"
        folder_index_str = tf.strings.split(folder_name, '.')[0]
        # Convert to int, subtract 1 to get 0-based label
        label = tf.strings.to_number(folder_index_str, out_type=tf.int32) - 1

        return {"image": image, "label": label}

    def image_preprocess_train(input_dict):
        """
        Applies data augmentation to 'input_dict["image"]' and
        adds 'concepts' from class_concepts using 'input_dict["label"]'.
        """
        image = input_dict['image']
        label = input_dict['label']
        image = tf.cast(image, tf.float32)

        # The random augmentation logic from your code:
        seed = tf.random.uniform((2,), 0, 1000000, dtype=tf.int32)
        scale = tf.random.uniform((), 0.08, 1.0, dtype=tf.float32)
        image_shape = tf.shape(image)
        area = tf.cast(image_shape[0] * image_shape[1], dtype=tf.float32)
        rescaled_area = area * scale
        aspect_ratio = tf.random.uniform((), 0.75, 1.333, dtype=tf.float32)
        new_aspect_ratio = aspect_ratio * tf.cast(image_shape[0], tf.float32) / tf.cast(image_shape[1], tf.float32)
        desired_size = (tf.math.sqrt(rescaled_area * new_aspect_ratio),
                        tf.math.sqrt(rescaled_area / new_aspect_ratio))
        new_size = [tf.minimum(tf.cast(desired_size[0], tf.int32), image_shape[0]),
                    tf.minimum(tf.cast(desired_size[1], tf.int32), image_shape[1]),
                    3]
        image = tf.image.stateless_random_crop(image, new_size, seed=seed)
        image = tf.image.resize_with_pad(image, tf.cast(desired_size[0], tf.int32), tf.cast(desired_size[1], tf.int32))
        image = tf.image.resize(image, [image_size, image_size])
        image = tf.image.stateless_random_flip_left_right(image, seed=seed)
        image = tf.image.stateless_random_brightness(image, max_delta=32. / 255., seed=seed)
        image = tf.image.stateless_random_saturation(image, lower=0.5, upper=1.5, seed=seed)

        # Scale to [-1, 1]
        image = (image / 127.5) - 1.
        image = tf.clip_by_value(image, -1, 1)

        # Build output
        output = {}
        output['data'] = image
        output['label'] = label
        # Gather concept vector for this label
        output['concepts'] = class_concepts[label, :]
        return output

    def image_preprocess_test(input_dict):
        """
        Simple center-crop (or pad) + resize for test images, no heavy augmentation.
        """
        image = input_dict['image']
        label = input_dict['label']
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        image = (image / 127.5) - 1.
        image = tf.clip_by_value(image, -1, 1)

        output = {}
        output['data'] = image
        output['label'] = label
        output['concepts'] = class_concepts[label, :]
        return output

    # List all JPG files in CUB_200_2011/images/*/*.jpg
    # If you truly want to replicate the official train/test split,
    # you'd parse "train_test_split.txt" instead. For now, we just do a naive approach:
    if train:
        # Shuffle for training
        ds = tf.data.Dataset.list_files("CUB_200_2011/images/*/*.jpg", shuffle=True)
    else:
        # No shuffle for test
        ds = tf.data.Dataset.list_files("CUB_200_2011/images/*/*.jpg", shuffle=False)

    # Map from file path -> {"image": decoded_image, "label": label}
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply augmentation or test preprocessing
    if augmented:
        ds = ds.map(image_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(image_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)

    return ds