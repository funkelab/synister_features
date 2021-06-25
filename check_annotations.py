import numpy as np
import skimage.measure
import zarr

dataset = '20210625'
original_dataset = '../data/source_data'
annotators = ['c0', 'c1', 'c2']
max_num_chunks = 20
layer_names = ['vesicles', 'cleft', 'cleft_membrane', 'cytosol', 'posts', 't-bars']

background_width = 50  # the space between individual synapses in the source data


def check_chunk(zarr_file, chunk_group):

    print(f"Checking annotations in {chunk_group}...")

    for synapse in range(10):
        check_synapse(zarr_file, f'{chunk_group}/{synapse}')


def check_synapse(zarr_file, synapse_group):

    find_empty_layers(zarr_file, synapse_group)
    find_non_unique_layers(zarr_file, synapse_group)
    find_dust(zarr_file, synapse_group)
    find_excess_labels(zarr_file, synapse_group)
    compare_intensities(zarr_file, synapse_group)


def find_empty_layers(zarr_file, synapse_group):

    empty_layers = []

    for layer_name in layer_names:

        ds_name = f'{synapse_group}/{layer_name}'

        layer = zarr_file[ds_name][:]
        if not has_annotations(layer):
            empty_layers.append(layer_name)

    if empty_layers:
        print(f"{synapse_group}: no annotations in layers {empty_layers}")


def find_non_unique_layers(zarr_file, synapse_group):

    non_unique_layers = []

    for layer_name in ['vesicles', 'posts']:

        ds_name = f'{synapse_group}/{layer_name}'

        layer = zarr_file[ds_name][:]
        if not has_unique_connected_components(layer):
            non_unique_layers.append(layer_name)

    if non_unique_layers:
        print(f"{synapse_group}: non-unique IDs in layers {non_unique_layers}")


def find_dust(zarr_file, synapse_group, max_size=10):
    '''Check that there are no small, accidental annotations (which we call
    "dust") that are at most `max_size` voxels big.'''

    dust_layers = []

    for layer_name in layer_names:

        ds_name = f'{synapse_group}/{layer_name}'

        layer = zarr_file[ds_name][:]
        if has_dust(layer, max_size):
            dust_layers.append(layer_name)

    if dust_layers:
        print(f"{synapse_group}: dust in layers {dust_layers}")


def find_excess_labels(zarr_file, synapse_group):

    excess_layers = []

    for layer_name in ['cleft', 'cleft_membrane', 'cytosol', 't-bars']:

        ds_name = f'{synapse_group}/{layer_name}'

        layer = zarr_file[ds_name][:]
        num_labels = count_labels(layer)
        if num_labels is None:
            print(f"{synapse_group}: no 0 label in layer {layer_name}!")
            continue

        if num_labels > 1:
            excess_layers.append(layer_name)

    if excess_layers:
        print(f"{synapse_group}: more than one label in layers {excess_layers}")


def compare_intensities(zarr_file, synapse_group):

    # synapse_group: synapses_c0_0/0
    #   split into: chunk_group / synapse_number

    chunk_group, synapse = synapse_group.split('/')
    synapse = int(synapse)

    raw = zarr_file[f'{synapse_group}/raw'][:]

    # find the correct chunk in source data
    chunk_filename = f'{original_dataset}/{chunk_group}.zarr'
    source_chunk_raw = zarr.open(chunk_filename, 'r')['raw']

    # cut out the relevant part for our synapse

    layer_width = source_chunk_raw.shape[2]
    synapse_width = (layer_width - (11 * background_width))//10
    start_x = background_width + synapse * (synapse_width + background_width)
    end_x = start_x + synapse_width

    source_raw = source_chunk_raw[:,:,start_x:end_x]

    # compare that they are equal

    is_equal = np.all(raw == source_raw)

    if not is_equal:
        print(f"{synapse_group}: raw data does not match source data")


def has_unique_connected_components(layer):
    '''This function checks whether each connected component in the given numpy
    array has a unique ID.'''

    ### Step 1: generate a binary mask ###

    unique_labels, label_counts = np.unique(layer, return_counts=True)
    nonzero_unique_labels = unique_labels[unique_labels!=0]

    for label in nonzero_unique_labels:

        binary_mask = layer == label

        ### Step 2: connected-component labeling ###
        # obtain a labeled array s.t. all the connected regions have the same integer value

        cc_labels = skimage.measure.label(binary_mask, connectivity=1)

        unique_cc_labels = np.unique(cc_labels)
        # unique_cc_labels = [0, 1, ...? ]
        #  if only one CC: unique_cc_labels = [0, 1]
        #  if two CCs:     unique_cc_labels = [0, 1, 2]
        #  if three CCs:   unique_cc_labels = [0, 1, 2, 3]
        #  ...and so on

        if unique_cc_labels.size != 2:

            print(f"Found {unique_cc_labels.size - 1} connected components "
                  "with the same ID!")
            return False

    return True


def has_dust(layer, max_size):

    _, label_counts = np.unique(layer, return_counts=True)

    return np.any(label_counts <= max_size)


def count_labels(layer):

    labels = np.unique(layer)

    if 0 not in labels:
        print(f"array {labels} does not contain 0")
        return None

    return labels.size - 1


def has_annotations(layer):

    return np.unique(layer).size > 1


if __name__ == "__main__":

    zarr_file = zarr.open(f'../data/{dataset}.zarr', 'r')

    for annotator in annotators:

        for chunk in range(max_num_chunks):

            chunk_group = f'synapses_{annotator}_{chunk}'

            if chunk_group not in zarr_file:
                continue

            check_chunk(zarr_file, chunk_group)
