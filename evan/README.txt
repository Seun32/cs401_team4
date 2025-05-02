Steps to generate heatmap:
1. Install CUB dataset
2. Prepare the attributes by running attribute_processing.py
3. Train model using main.py which creates a saved_model directory
4. Reconfigure saved_model with reexport_with_signature.py which will make a signed_model directory
5. Generate heatmaps with feature_attribution_tf.py which will generate heatmaps in expl_plt directory

- There are examples of the generated heatmaps in the current expl_plt directory
- Bug fixes have been applied to the model training files, which differentiate it from the Addressing Concept Leakage model
- Heatmaps highlight the top 500 contributing pixels per image.
