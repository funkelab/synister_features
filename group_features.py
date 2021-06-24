import numpy as np
import json
import copy

dataset = '20210609'

def get_skipped_synapses(features):

    skipped_synapses = []
    feature_names = ['cleft_mean_intensity', 'cleft_membrane_mean_intensity',
    't-bars_mean_intensity', 'cytosol_mean_intensity',
    'cleft_median_intensity', 'cleft_membrane_median_intensity',
    't-bars_median_intensity','cytosol_median_intensity', 'post_count',
    'num_vesicles', 'vesicle_sizes', 'vesicle_circularities']

    for synapse in features:

        for feature_name in list(synapse.keys())[5:]:

            if feature_name.split('_')[-1] == 'intensity':
                if synapse[feature_name] != None: break

            elif feature_name in ['post_count', 'num_vesicles']:
                if synapse[feature_name]!=0: break

            elif feature_name.split('_')[0] == 'vesicle':
                if len(synapse[feature_name])!=0: break
                else:
                    skipped_synapses.append(synapse)
                    break
    return skipped_synapses

def filter_features_by_condition(features, condition):

    feature_name = condition[1]
    feature_short_name = feature_name.split('_')[0]
    skipped_synapses = get_skipped_synapses(features)

    if feature_short_name == 'cleft' or feature_short_name == 't-bars':

        mean_or_median = feature_name.split('_')[-2]
        features_filtered = [synapse for synapse in features if
                      isinstance(synapse[f'cytosol_{mean_or_median}_intensity'],float) &
                      isinstance(synapse[f'cleft_membrane_{mean_or_median}_intensity'],float) &
                      isinstance(synapse[f'{feature_short_name}_{mean_or_median}_intensity'],float) ]

    elif feature_name in ['post_count', 'num_vesicles']:
        features_filtered = [synapse for synapse in features if
                      synapse not in skipped_synapses]

    elif feature_name.split('_')[0] == 'vesicle':
        features_filtered = [synapse for synapse in features if
                len(synapse[feature_name])!=0]
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

if __name__ == "__main__":

    # handle full statistics
    with open(f"synapse_statistics_{dataset}.json", 'r') as f:
        statistics = json.load(f)

#     # handle duplicates
#     with open(f"duplicate_statistics_{dataset}.json", 'r') as f:
#         statistics = json.load(f)

    feature_names = ['cleft_mean_intensity', 't-bars_mean_intensity',
            't-bars_mean_intensity', 'cleft_median_intensity',
            't-bars_median_intensity', 'post_count', 'num_vesicles',
            'vesicle_sizes', 'vesicle_circularities']

    extracted_features = {}

    # conditions = [(by_nt_type, feature_name)]
    # e.g. conditions = [('by_nt_types', 'vesicle_size'), ('by_annotators',
    # 'post_count'),...]

    conditions_by_nt_types = list(zip(['by_nt_types']*len(feature_names), feature_names))
    conditions_by_annotators = list(zip(['by_annotators']*len(feature_names), feature_names))
    conditions = conditions_by_nt_types + conditions_by_annotators
    print(f'conditions are {conditions}')

    for condition in conditions:

        print(f'Extracting feature under condition {condition}')
        statistics_filtered = filter_statistics_by_condition(statistics, condition)

        # extracted_features = {'gaba':[..],'glutamate':[..], 'acetylcholine':[..]} 
        # or {'c0':[..],'c1':[..], 'c2':[..]}
        extracted_feature = extract_feature(statistics_filtered, condition)
        extracted_features.update({str(condition) : extracted_feature})

    # handle full statistics
    with open(f'extracted_features_{dataset}.json', 'w') as f:
        json.dump(extracted_features, f, indent=2)

#     # handle duplicates
#     with open(f'extracted_duplicate_features_{dataset}.json', 'w') as f:
#         json.dump(extracted_features, f, indent=2)


