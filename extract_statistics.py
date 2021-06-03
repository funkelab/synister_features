import numpy as np
import skimage.measure
import zarr
import json
from skimage.morphology import flood

dataset = '20210602'
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
    average_intensities = get_average_intensities(zarr_file, synapse_group)
    post_account = get_post_account(zarr_file, synapse_group)

    synapse_stats = {
        'annotator': annotator,
        'chunk_number': chunk_number,
        'synapse_number': synapse,
        'synapse_id': synapse_id,
        'neurotransmitter': neurotransmitter,
        'average_intensities': average_intensities,
        'post_account': post_account
    }

    # get synapse statistics

    synapse_stats.update(extract_vesicle_sizes(zarr_file, synapse_group))

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

def get_average_intensities(zarr_file, synapse_group):

    layer_names = ['cleft', 'cleft_membrane', 'cytosol', 't-bars']
    average_intensities = {}

    for layer in layer_names:

        # get raw intensities and annotated regions
        raw = zarr_file[f'{synapse_group}/raw'][:]
        # create a mask for the background region
        mask = flood(zarr_file[f'{synapse_group}/{layer}'][:], (0,0,0), tolerance=0)

        # set the intensities of the backgournd region to be zero
        raw[mask] = 0
        average_intensities.update({f'{layer}_average_intensities' : np.average(raw[raw!=0])})

    print(f"{average_intensities}")

    return average_intensities

def get_post_account(zarr_file, synapse_group):

    layer = zarr_file[f'{synapse_group}/posts'][:]
    unique_labels, label_counts = np.unique(layer, return_counts=True)
    print(f"post account {len(label_counts) - 1}")
    return len(label_counts) - 1

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

    with open('synapse_statistics.json', 'w') as f:
        json.dump(synapse_stats, f, indent=2)
