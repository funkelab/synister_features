import numpy as np
import json
import copy

dataset = '20210625'

def filter_features_by_condition(features, condition, duplicate_number):

    feature_name = condition[1]
    feature_short_name = feature_name.split('_')[0]

    # prefilter features according to duplicate_number

    if duplicate_number == 1 or duplicate_number > 2:

        prefiltered_features = [synapse for synapse in features if
                synapse['duplicate_number'] == duplicate_number]

    elif duplicate_number == None:
        prefiltered_features = features

    else:
        duplicate_synapse_ids = get_duplicate_synapse_ids(features)
        prefiltered_features  = [synapse for synapse in features if
                synapse['synapse_id'] in duplicate_synapse_ids]

    # filter again based on condition

    if feature_short_name == 'cleft' or feature_short_name == 't-bars':

        mean_or_median = feature_name.split('_')[-2]
        features_filtered = [synapse for synapse in prefiltered_features if
                      isinstance(synapse[f'cytosol_{mean_or_median}_intensity'],float) &
                      isinstance(synapse[f'cleft_membrane_{mean_or_median}_intensity'],float) &
                      isinstance(synapse[f'{feature_short_name}_{mean_or_median}_intensity'],float) ]

    elif feature_short_name == 'vesicle':
        features_filtered = [synapse for synapse in prefiltered_features if
                len(synapse[feature_name])!=0]

    elif feature_name in ['num_vesicles', 'post_count']:
        features_filtered = prefiltered_features

    else:
        print(f'ERROR: feature name {feature_name} does not exist')
        return None

    return features_filtered

def extract_feature(features, condition):

    feature_name = condition[1].split('_')[0]

    if feature_name == 'cleft' or feature_name == 't-bars':

        feature_types,features_by_types,_ = catalog_features_by_condition(features, condition)
        feature_values = []

        for synapse in features:
            mean_or_median = condition[1].split('_')[-2]
            val = synapse[f'{feature_name}_{mean_or_median}_intensity']
            minimum = synapse[f'cleft_membrane_{mean_or_median}_intensity']
            maximum = synapse[f'cytosol_{mean_or_median}_intensity']
            feature_values.append((val-minimum)/(maximum-minimum))

        for feature_type, feature_value in zip(feature_types, feature_values):
            features_by_types[feature_type].append(feature_value)

    else:
        feature_types, features_by_types, feature_values = catalog_features_by_condition(features, condition)

        for feature_type, feature_value in zip(feature_types, feature_values):
            features_by_types[feature_type].append(feature_value)

    return features_by_types

def get_duplicate_synapse_ids(features):

    duplicate_sets = {}
    duplicate_synapse_ids = set({})

    for i, synapse in enumerate(features):

        synapse_id = synapse['synapse_id']
        if synapse_id in duplicate_sets:
            duplicate_sets[synapse_id].append(i)
        else:
            duplicate_sets[synapse_id] = [i]

    for synapse_id in duplicate_sets.keys():

        if len(duplicate_sets[synapse_id]) > 1:
            duplicate_synapse_ids.add(synapse_id)

    return duplicate_synapse_ids

def catalog_features_by_condition(features, condition):

    if condition[0] == 'by_nt_types':

        features_by_types = {'gaba':[],'glutamate':[], 'acetylcholine':[]}

        if condition[1].split('_')[0] == 'vesicle':
            feature_values = [size for synapse in features for size in
                    synapse[condition[1]] ]
            feature_types = [typ for synapse in features for typ in
                    [synapse['neurotransmitter']]*len(synapse[condition[1]]) ]

        else:
            feature_values = [synapse[condition[1]] for synapse in features]
            feature_types = [synapse['neurotransmitter'] for synapse in features]

    if condition[0] == 'by_annotators':

        features_by_types = {'c0':[],'c1':[], 'c2':[]}

        if condition[1].split('_')[0] == 'vesicle':
            feature_values = [size for synapse in features for size in
                    synapse[condition[1]] ]
            feature_types = [typ for synapse in features for typ in
                    [synapse['annotator']]*len(synapse[condition[1]]) ]

        else:
            feature_values = [synapse[condition[1]] for synapse in features]
            feature_types = [synapse['annotator'] for synapse in features]

    return feature_types, features_by_types, feature_values

def group_features_by_conditions(duplicate_number):
    '''
    Group synapse features by different conditions.

    A "condition" is a tuple of two parts:

        1. what to group by (either by neurotransmitter or by annotator)
        2. the name of a feature (e.g., "num_vesicles")

    Returns a dictionary that looks like:

    ```
    {
        ('by_nt_types', 'feature'): {
            'gaba': ...,
            'glutamate': ...,
            'acetylcholine': ...
        },
        # ... more features

        ('by_annotators', 'feature'): {
            'c0': ...,
            'c1': ...,
            'c2': ...
        },
        # ... more features
    }
    ```

    Features should have been extracted earlier with `./extract_features.py`,
    which puts them into <synapse_features_{dataset}.json>.


    This function can be used in a jupyter notebook:

    ```
    from group_features import group_features_by_conditions

    features = group_features_by_conditions()
    ```
    '''

    # handle full features
    with open(f"synapse_features_{dataset}.json", 'r') as f:
        features = json.load(f)

    feature_names = ['cleft_mean_intensity', 't-bars_mean_intensity',
            't-bars_mean_intensity', 'cleft_median_intensity',
            't-bars_median_intensity', 'post_count', 'num_vesicles',
            'vesicle_sizes', 'vesicle_circularities']

    grouped_features = {}

    conditions_by_nt_types = list(zip(['by_nt_types']*len(feature_names), feature_names))
    conditions_by_annotators = list(zip(['by_annotators']*len(feature_names), feature_names))
    conditions = conditions_by_nt_types + conditions_by_annotators
    print(f'conditions are {conditions}')

    # 'None' duplicate_number means to group only duplicate features
    for condition in conditions:

        print(f'Extracting feature under condition {condition}')
        features_filtered = filter_features_by_condition(
            features,
            condition,
            duplicate_number)

        extracted_feature = extract_feature(features_filtered, condition)
        grouped_features.update({str(condition) : extracted_feature})

    return grouped_features

