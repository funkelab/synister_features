import numpy as np
import skimage.measure
import zarr
import json
import math
import sys

dataset = '20210609'
file_to_ids_json = "../data/source_data/file_to_ids.json"
ids_to_nt_json = "../data/source_data/ids_to_nt.json"
annotators = ['c0', 'c1', 'c2']
max_num_chunks = 20

file_to_ids = None
ids_to_nt = None

def process_chunk(zarr_file, chunk_group):

    chunk_stats = []

    for synapse in range(10):
        synapse_stats = process_synapse(zarr_file, f'{chunk_group}/{synapse}', synapse)
        chunk_stats.append(synapse_stats)

    return chunk_stats

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
    mean_intensities = get_mean_intensities(zarr_file, synapse_group)
    median_intensities = get_median_intensities(zarr_file, synapse_group)
    post_count = get_post_count(zarr_file, synapse_group)

    synapse_stats = {
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

    # get synapse statistics
    synapse_stats.update(extract_vesicle_sizes(zarr_file, synapse_group))
    synapse_stats.update(extract_vesicle_circularities(zarr_file, synapse_group))

    return synapse_stats

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

def extract_vesicle_circularities(zarr_file, synapse_group):

    vesicle_circularities = []

    ds_name = f"{synapse_group}/{'vesicles'}"
    layer = zarr_file[ds_name][:]

    # get the layer with annotations
    annotated_layer = get_annotated_layer(layer)

    if annotated_layer == 'skip':
        return {'vesicle_circularities': []}

    # generate a binary mask
    unique_labels, label_counts = np.unique(annotated_layer, return_counts=True)
    nonzero_unique_labels = unique_labels[unique_labels!=0]

    for label in nonzero_unique_labels:

        binary_mask = annotated_layer == label

        cc_labels = skimage.measure.label(binary_mask, connectivity=1)
        properties = skimage.measure.regionprops(cc_labels)
        vesicle_circularity = (4 * math.pi *
                properties[0]['area'])/(properties[0]['perimeter']**2)

        print(f'vc = {vesicle_circularity}')
        vesicle_circularities.append(vesicle_circularity)

    return {
            'vesicle_circularities': vesicle_circularities
            }

def get_mean_intensities(zarr_file, synapse_group):

    layer_names = ['cleft', 'cleft_membrane', 'cytosol', 't-bars']
    mean_intensities = {}

    # get raw intensities and annotated regions
    raw = zarr_file[f'{synapse_group}/raw'][:]

    for layer_name in layer_names:

        layer = zarr_file[f'{synapse_group}/{layer_name}'][:]

        if np.sum(layer) == 0:
            mean_intensities.update({
                f'{layer_name}': 'skip'
            })

        else:
            mean_intensities.update({
                f'{layer_name}': float(np.average(raw[layer!=0]))
            })

    if (mean_intensities['cleft'] != 'skip') & (mean_intensities['cleft_membrane'] != 'skip'):

        cleft_membrane = zarr_file[f'{synapse_group}/cleft_membrane'][:]
        cleft = zarr_file[f'{synapse_group}/cleft'][:]

        mask_cleft = cleft != 0
        mask_cleft_membrane = cleft_membrane != 0

        # this is True where membrane == True AND not cleft == True
        mask_only_cleft_membrane = mask_cleft_membrane & ~mask_cleft
        mean_intensities.update({
            f'cleft_membrane' :
            float(np.average(raw[mask_only_cleft_membrane]))
        })

    return mean_intensities

def get_median_intensities(zarr_file, synapse_group):

    layer_names = ['cleft',  'cleft_membrane', 'cytosol', 't-bars']
    median_intensities = {}

    # get raw intensities and annotated regions
    raw = zarr_file[f'{synapse_group}/raw'][:]

    for layer_name in layer_names:

        layer = zarr_file[f'{synapse_group}/{layer_name}'][:]

        median_intensities.update({
            f'{layer_name}': float(np.median(raw[layer!=0]))
        })

        if np.sum(layer) == 0:
            median_intensities.update({
                f'{layer_name}': 'skip'
            })
        else:
            median_intensities.update({
                f'{layer_name}': float(np.median(raw[layer!=0]))
            })
    if (median_intensities['cleft'] != 'skip') & (median_intensities['cleft_membrane'] != 'skip'):

        cleft_membrane = zarr_file[f'{synapse_group}/cleft_membrane'][:]
        cleft = zarr_file[f'{synapse_group}/cleft'][:]

        mask_cleft = cleft != 0
        mask_cleft_membrane = cleft_membrane != 0

        # this is True where membrane == True AND not cleft == True
        mask_only_cleft_membrane = mask_cleft_membrane & ~mask_cleft
        median_intensities.update({
            f'cleft_membrane' :
            float(np.median(raw[mask_only_cleft_membrane]))
        })

    return median_intensities

def get_post_count(zarr_file, synapse_group):

    layer = zarr_file[f'{synapse_group}/posts'][:]
    unique_labels, label_counts = np.unique(layer, return_counts=True)
    # print(f"post count {len(label_counts) - 1}")
    return len(label_counts) - 1

def get_annotated_layer(layer):

    for z in range(29):
        if np.sum(layer[z,:,:])!=0:
            print(f'annotated layer is z = {z}')
            return layer[z,:,:].reshape(-1, layer[z,:,:].shape[-1])

    # no annotation found in all layers
    return 'skip'

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

    synapse_stats = []

    for annotator in annotators:

        for chunk in range(max_num_chunks):

            chunk_group = f'synapses_{annotator}_{chunk}'

            if chunk_group not in zarr_file:
                continue

            print(f"Processing chunk {chunk_group}...")

            synapse_stats += process_chunk(zarr_file, chunk_group)

    with open(f'synapse_statistics_{dataset}.json', 'w') as f:
        json.dump(synapse_stats, f, indent=2)
