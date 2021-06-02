import os
import skimage.io
import numpy as np
import skimage.measure
import zarr
import zlib


def check_annotation(layer):

    unique_labels = np.unique(layer, return_counts=False)
    
    # print(f"unique labels are {unique_labels}")
    
    if unique_labels.size > 1:

        return True
    
    else:

        return False

    
if __name__ == "__main__":

    chunk_label = 'c0_0'

    for synapse_number in range(10):

        ds_path = f'/groups/funke/home/luk/synister/run/data/20210525.zarr/synapses_{chunk_label}/{synapse_number}'

        zarr_file = zarr.open(ds_path, 'r')

        vesicle_layers = zarr_file['vesicles'][:]
        cleft_layers = zarr_file['cleft'][:]
        cleft_membrane_layers = zarr_file['cleft_membrane'][:]
        cytosol_layers = zarr_file['cytosol'][:]
        posts_layers = zarr_file['posts'][:]
        t_bars_layers = zarr_file['t-bars'][:]
       
        all_layers = np.array([vesicle_layers, cleft_layers, cleft_membrane_layers, cytosol_layers, posts_layers, t_bars_layers])  
        # all_layers = np.array([vesicle_layers])

        for layer in all_layers:
            
            if not(check_annotation(layer)):

                print(f"synapses_{chunk_label}/{synapse_number} NOT ANNOTATED")
