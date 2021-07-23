import numpy as np
import json
import copy

dataset = '20210722'


def filter_synapses(features, feature_name):
    '''Filter out all synapses that do not have the requested feature.'''

    if feature_name.startswith('vesicle'):

        features_filtered = [
            synapse
            for synapse in features
            if synapse[feature_name]
        ]

    else:

        features_filtered = [
            synapse
            for synapse in features
            if synapse[feature_name] is not None
        ]

    return features_filtered


def get_duplicate_synapse_ids(features):
    '''Return a set of synapse IDs that have been annotated more than once.'''

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


def group_features(features, feature_name, group_condition):
    '''Group features with a given name by the given condition.'''

    # make sure to not use undefined features
    features = filter_synapses(features, feature_name)

    grouping_keys = [
        {
            'by_annotators': 'annotator',
            'by_nt_types': 'neurotransmitter'
        }[c]
        for c in group_condition
    ]

    # list of features for vesicles, instead of a single number
    if feature_name.startswith('vesicle'):

        feature_values = [
            value
            for synapse in features
            for value in synapse[feature_name]
        ]
        conditions = [
            condition
            for synapse in features
            for condition in list(
            ((tuple([synapse[key] for key in grouping_keys]),)
            *len(synapse[feature_name]) )
            )
        ]

    else:

        feature_values = [synapse[feature_name] for synapse in features]
        conditions = []
        for synapse in features:
            condition = []
            for key in grouping_keys:
                condition.append(synapse[key])
            conditions.append(tuple(condition))

        #for key in grouping_keys:
        #    for synapse in features:
        #        condition = tuple([synapse[key]])
        #        conditions.append(tuple(
        #            [synapse[key]]
        #        ))
        #conditions = [
        #    tuple([synapse[key]
        #    for synapse in features
        #    for key in grouping_keys])
        #]

    grouped_features = {}
    for condition, feature_value in zip(conditions, feature_values):
        if condition not in grouped_features:
            grouped_features[condition] = []
        grouped_features[condition].append(feature_value)

    # print('grouped_features: ', '\n', f'{grouped_features}')
    return grouped_features

def group_features_by_conditions(condition, filter='unique'):
    '''
    Group synapse features by different conditions.

    Args:

        condition (tuple of strings):

            Possible values are "by_nt_types" and "by_annotators".

        filter (string, optional):

            Possible values are "unique", "same", "all":

                "unique": All synapses with duplicate_number 1 (default).

                "same": Only synapses that have been annotated by at least two
                annotators. Results will be grouped by pairs of annotators.

                "all": All synpases (including duplicates).


    Returns a dictionary that looks like:

    ```
    {
        <feature_name>: {
            <condidation_1>: ...,
            <condidation_2>: ...,
            <condidation_3>: ...
            ...
        },
        # ... more features
    }
    ```

    Examples:

    ```
    group_features_by_conditions(('by_annotators',))
    ```
    Returns:
    ```
    {
        'vesicle_sizes': {
            ('c0',): [....],
            ('c1',): [....],
            ('c2',): [....]
        }
    }
    ```

    ```
    group_features_by_conditions(('by_annotators', 'by_nt_types'))
    ```
    Returns:
    ```
             ...    {
        'vesicle_sizes': {
            ('c0', 'glutamate'): [....],
            ('c0', 'gaba'): [....],
            ...
            ('c1', 'glutamate'): [....],
            ('c1', 'gaba'): [....]
            ...
        }
    }
    ```

    Features should have been extracted earlier with `./extract_features.py`,
    which puts them into <synapse_features_{dataset}.json>.


    This function can be used in a jupyter notebook:

    ```
    from group_features import group_features_by_conditions

    features = group_features_by_conditions(('by_nt_types',))
    ```
    '''

    # handle full features
    with open(f"synapse_features_{dataset}.json", 'r') as f:
        features = json.load(f)

    feature_names = ['cleft_mean_intensity', 't-bars_mean_intensity',
            't-bars_mean_intensity', 'cleft_median_intensity',
            't-bars_median_intensity', 't-bars_mean_normalized_intensity',
            'cleft_mean_normalized_intensity',
            't-bars_median_normalized_intensity',
            'cleft_median_normalized_intensity',
            'post_count', 'num_vesicles',
            'vesicle_sizes', 'vesicle_eccentricities']

    # filter based on duplicate numbers

    if filter == 'unique':
        features = [
            f
            for f in features
            if f['duplicate_number'] == 1
        ]
    elif filter == 'same':
        duplicate_synapse_ids = get_duplicate_synapse_ids(features)
        features = [
            f
            for f in features
            if f['synapse_id'] in duplicate_synapse_ids
        ]
    elif filter == 'all':
        pass
    else:
        raise RuntimeError("'filter' should be 'unique', 'same', or 'all'")

    for c in condition:
        assert c in ['by_nt_types', 'by_annotators']

    grouped_features = {
        feature_name: group_features(features, feature_name, condition)
        for feature_name in feature_names
    }

    return grouped_features

