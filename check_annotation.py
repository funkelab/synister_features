import os
import skimage.io
import numpy as np
import skimage.measure
import zarr
import zlib


dataset = '20210525'
annotators = ['c0', 'c1', 'c2']
max_num_chunks = 20
layer_names = ['vesicles', 'cleft', 'cleft_membrane', 'cytosol', 'posts', 't-bars']


def has_annotations(layer):

    return np.unique(layer).size > 1


def check_chunk(zarr_file, chunk_group):

    print(f"Checking annotations in {chunk_group}...")

    for synapse in range(10):
        check_synapse(zarr_file, f'{chunk_group}/{synapse}')


def check_synapse(zarr_file, synapse_group):

    find_empty_layers(zarr_file, synapse_group)


def find_empty_layers(zarr_file, synapse_group):

    empty_layers = []

    for layer_name in layer_names:

        ds_name = f'{synapse_group}/{layer_name}'

        layer = zarr_file[ds_name][:]
        if not has_annotations(layer):
            empty_layers.append(layer_name)

    if empty_layers:
        print(f"{synapse_group}: no annotations in layers {empty_layers}")


if __name__ == "__main__":

    zarr_file = zarr.open(f'../data/{dataset}.zarr', 'r')

    for annotator in annotators:

        for chunk in range(max_num_chunks):

            chunk_group = f'synapses_{annotator}_{chunk}'

            if chunk_group not in zarr_file:
                continue

            check_chunk(zarr_file, chunk_group)
