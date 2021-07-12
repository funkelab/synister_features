Analysis Pipeline
=================

The analysis has several steps: first, we extract all features that we want to
analyze, then we group, analyze, and plot them.

1. Extract Features:
----------------------

  Run `./extract_features.py`

  The name of the dataset to use is hard-coded at the top of this file.

  This will create a `synapse_features_<dataset name>.json` JSON file.

  The output JSON looks like this:

  ```
  [
    {
      "annotator": <annotator ID>,
      "chunk_number": ...,
      "synapse_number": ...,
      "synapse_id": ...,
      "neurotransmitter": ...,
      <list of feature names, mapping to values>
    }
  ]
  ```

  List of feature names is:

    "cleft_mean_intensity"
    "cleft_median_intensity"
    "cleft_membrane_mean_intensity"
    "cleft_membrane_median_intensity"
    "cytosol_mean_intensity"
    "cytosol_median_intensity"
    "num_vesicles"
    "post_count"
    "t-bars_mean_intensity"
    "t-bars_median_intensity"
    "vesicle_circularities"
    "vesicle_sizes"

  There are a few special cases:

    1. A synapse was not annotated at all.
      ⇒ this synapse will not be included in the JSON

    2. A feature might not be present (e.g., there is no cleft, can't measure cleft intensity)
      ⇒ this feature will be set to `None`

    3. A synapse is a duplicate of another synapse (two or more annotators did the same)
      ⇒ for each synapse (duplicate or not) we store a `duplicate_precedence` "feature"
        for analysis, we would only use those synapses with `duplicate_precedence==1`

        TODO: for each set of duplicates, randomly assign `duplicate_precedence`
        TODO: make sure that `duplicate_precedence` is always the same (but random)

      With that, we can easily filter for all unique synapses with:

        ```
        filtered_synapses = [
          synapse
          for synapse in synapses
          if synapse['duplicate_precedence'] == 1
        ]
        ```

2. Group, Analyze, and Visualize
--------------------------------

  `group_features.py` contains functions to read and group features from the
  JSON of step 1.

