import numpy as np
import skimage.measure
import zarr
import json
import math
import sys
import random

dataset = '20210630'
file_to_ids_json = "../data/source_data/file_to_ids.json"
ids_to_nt_json = "../data/source_data/ids_to_nt.json"
annotators = ['c0', 'c1', 'c2']
max_num_chunks = 20

file_to_ids = None
ids_to_nt = None

def process_chunk(zarr_file, chunk_group):

    chunk_stats = []

    for synapse in range(10):

        synapse_group = f'{chunk_group}/{synapse}'

        # not to process synapses that are skipped
        if not skip_synapse(zarr_file, synapse_group):
            synapse_features = process_synapse(zarr_file, synapse_group, synapse)
            chunk_stats.append(synapse_features)

    return chunk_stats

def skip_synapse(zarr_file, synapse_group):

    layer_names = ['cleft', 'cleft_membrane', 'cytosol', 'posts',
            't-bars', 'vesicles']

    for layer_name in layer_names:

        ds_name = f'{synapse_group}/{layer_name}'
        layer = zarr_file[ds_name][:]

        if np.sum(layer) != 0:
            # print(f'{ds_name} sum to {np.sum(layer)}')
            return False

    print(f'skip synapse {ds_name}')
    return True

def process_synapse(zarr_file, synapse_group, synapse):

    # synapse_group: synapses_c0_0/0
    # split synapse_group by /
    #   chunk_name / synapse
    # split chunk_name by _
    #   synapses _ annotator _ chunk_number

    chunk_name, _ = synapse_group.split('/')
    _, annotator, chunk_number = chunk_name.split('_')
    chunk_number = int(chunk_number)

    synapse_id = get_synapse_id(annotator, chunk_number, synapse)
    neurotransmitter = get_neurotransmitter(synapse_id)
    mean_intensities = agglomerate_intensities(
        zarr_file,
        synapse_group,
        np.mean)
    median_intensities = agglomerate_intensities(
        zarr_file,
        synapse_group,
        np.median)
    post_count = get_post_count(zarr_file, synapse_group)

    # feature_values.append((val-minimum)/(maximum-minimum))

    synapse_features = {
        'annotator': annotator,
        'chunk_number': chunk_number,
        'synapse_number': synapse,
        'synapse_id': synapse_id,
        'neurotransmitter': neurotransmitter,
        'cleft_mean_intensity': mean_intensities['cleft'],
        'cleft_membrane_mean_intensity': mean_intensities['cleft_membrane'],
        't-bars_mean_intensity': mean_intensities['t-bars'],
        'cytosol_mean_intensity': mean_intensities['cytosol'],
        'cleft_median_intensity': median_intensities['cleft'],
        'cleft_membrane_median_intensity': median_intensities['cleft_membrane'],
        't-bars_median_intensity': median_intensities['t-bars'],
        'cytosol_median_intensity': median_intensities['cytosol'],
        'post_count': post_count
    }

    # add normalized features

    for agglo in ['mean', 'median']:
        for structure in ['t-bars', 'cleft']:
            intensity = synapse_features[f'{structure}_{agglo}_intensity']
            if intensity is None:
                normalized_intensity = None
            else:
                minimum = synapse_features[f'cleft_membrane_{agglo}_intensity']
                maximum = synapse_features[f'cytosol_{agglo}_intensity']
                normalized_intensity = ((intensity - minimum)/(maximum - minimum))
            synapse_features[f'{structure}_{agglo}_normalized_intensity'] = \
                normalized_intensity

    # get synapse statistics
    synapse_features.update(extract_vesicle_sizes(zarr_file, synapse_group))
    synapse_features.update(extract_vesicle_eccentricities(zarr_file, synapse_group))

    return synapse_features

def get_synapse_id(annotator, chunk_number, synapse_number):

    global file_to_ids

    if file_to_ids is None:
        with open(file_to_ids_json, 'r') as f:
            file_to_ids = json.load(f)

    tag = f'{annotator}_{chunk_number}'
    return file_to_ids[tag][synapse_number]

def get_neurotransmitter(synapse_id):

    global ids_to_nt

    if ids_to_nt is None:
        with open(ids_to_nt_json, 'r') as f:
            ids_to_nt_tmp = json.load(f)
            # Python json loads keys as strings:
            ids_to_nt = {int(k): v for k, v in ids_to_nt_tmp.items()}

    return ids_to_nt[synapse_id]

def extract_vesicle_sizes(zarr_file, synapse_group):

    vesicles = zarr_file[f'{synapse_group}/vesicles'][:]

    vesicle_ids, vesicle_sizes = np.unique(vesicles, return_counts=True)
    vesicle_sizes = list([int(s) for s in vesicle_sizes[vesicle_ids!=0]])

    return {
        'num_vesicles': len(vesicle_sizes),
        'vesicle_sizes': vesicle_sizes
    }

def extract_vesicle_eccentricities(zarr_file, synapse_group):

    vesicle_eccentricities = []

    ds_name = f"{synapse_group}/{'vesicles'}"
    layer = zarr_file[ds_name][:]

    # get the layer with annotations
    annotated_layer = get_annotated_layer(layer)

    if annotated_layer is None:
        return {'vesicle_eccentricities': []}

    # generate a binary mask
    unique_labels, label_counts = np.unique(annotated_layer, return_counts=True)
    nonzero_unique_labels = unique_labels[unique_labels!=0]

    for label in nonzero_unique_labels:

        binary_mask = annotated_layer == label

        cc_labels = skimage.measure.label(binary_mask, connectivity=1)
        properties = skimage.measure.regionprops(cc_labels)
        vesicle_eccentricity = properties[0]['eccentricity']

        vesicle_eccentricities.append(vesicle_eccentricity)

    return {
            'vesicle_eccentricities': vesicle_eccentricities
            }

def agglomerate_intensities(zarr_file, synapse_group, agglo_fun):

    layer_names = ['cleft', 'cleft_membrane', 'cytosol', 't-bars']
    agglomerated_intensities = {}

    # get raw intensities and annotated regions
    raw = zarr_file[f'{synapse_group}/raw'][:]

    for layer_name in layer_names:

        layer = zarr_file[f'{synapse_group}/{layer_name}'][:]

        if np.sum(layer) == 0:
            agglomerated_intensities.update({
                f'{layer_name}': None
            })

        else:
            agglomerated_intensities.update({
                f'{layer_name}': float(agglo_fun(raw[layer!=0]))
            })

    if (
            agglomerated_intensities['cleft'] != None and
            agglomerated_intensities['cleft_membrane'] != None):

        cleft_membrane = zarr_file[f'{synapse_group}/cleft_membrane'][:]
        cleft = zarr_file[f'{synapse_group}/cleft'][:]

        mask_cleft = cleft != 0
        mask_cleft_membrane = cleft_membrane != 0

        # this is True where membrane == True AND not cleft == True
        mask_only_cleft_membrane = mask_cleft_membrane & ~mask_cleft
        agglomerated_intensities.update({
            f'cleft_membrane' :
            float(agglo_fun(raw[mask_only_cleft_membrane]))
        })

    return agglomerated_intensities

def get_post_count(zarr_file, synapse_group):

    layer = zarr_file[f'{synapse_group}/posts'][:]
    unique_labels, label_counts = np.unique(layer, return_counts=True)

    return len(label_counts) - 1

def get_annotated_layer(layer):

    for z in range(29):
        if np.sum(layer[z,:,:])!=0:
            return layer[z,:,:].reshape(-1, layer[z,:,:].shape[-1])

    # no annotation found in all layers
    return None

def assign_number_to_duplicates(synapse_features):

    # dictionary from synapse_id to list of indices into synapse_features
    duplicate_sets = {}
    for i, synapse in enumerate(synapse_features):
        synapse_id = synapse['synapse_id']
        if synapse_id in duplicate_sets:
            duplicate_sets[synapse_id].append(i)
        else:
            duplicate_sets[synapse_id] = [i]

    random.seed(1976)  # Fei-Fei Li's birthyear :)

    # assign random numbers to each duplicate
    duplicate_numbers = {}
    for synapse_id, duplicate_set in duplicate_sets.items():

        # we want that: [2, 1] or [1, 2] ....
        # we do not want that: [1, 1] or [2, 2]
        duplicate_number = list(range(1, len(duplicate_set) + 1))
        random.shuffle(duplicate_number)
        duplicate_numbers[synapse_id] = duplicate_number

    for synapse_id in duplicate_sets.keys():

        indices = duplicate_sets[synapse_id]
        numbers = duplicate_numbers[synapse_id]

        assert len(indices) == len(numbers)

        for i, p in zip(indices, numbers):
            synapse_features[i]['duplicate_number'] = p

if __name__ == "__main__":

    zarr_file = zarr.open(f'../data/{dataset}.zarr', 'r')

    # what we want:
    #
    # list of dictionaries, one for each synapse, like:
    #
    #   {
    #       'annotator': 'c0',
    #       'chunk_number': 1,
    #       'synapse_number': 2,  # the number of the syn in the chunk
    #       'synapse_id': '38472171743',  # original CATMAID ID
    #       'neurotransmitter': 'gaba',
    #
    #       'num_vesicles': 5,
    #       'vesicle_sizes': [23, 43, ...]
    #          (...and a few more later)
    #   }

    synapse_features = []

    for annotator in annotators:

        for chunk in range(max_num_chunks):

            chunk_group = f'synapses_{annotator}_{chunk}'

            if chunk_group not in zarr_file:
                continue

            print(f"Processing chunk {chunk_group}...")

            synapse_features += process_chunk(zarr_file, chunk_group)

    assign_number_to_duplicates(synapse_features)

    with open(f'synapse_features_{dataset}.json', 'w') as f:
        json.dump(synapse_features, f, indent=2)
