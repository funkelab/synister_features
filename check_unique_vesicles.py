import os
import skimage.io
import numpy as np
import skimage.measure
import zarr
import zlib

def fake_nonunique_vesicles(annotated_vesicle_layer):

    '''create nonunique vesicles for testing purpose'''
    # TEST CASE 1: Change all the 52 (unique) to 51 (non-unique)
    # annotated_vesicle_layer[annotated_vesicle_layer!=0] = 51
    
    # TEST CASE 2: Change some of the 52 (unique) to 51 (non-unique)
    condition_1 = annotated_vesicle_layer == 52
    condition_2 = annotated_vesicle_layer == 46
    annotated_vesicle_layer[(condition_1)] = 54

    return annotated_vesicle_layer
    

def check_unique_vesicles(vesicle_layers):

### Step 1: generate a binary mask ###
    
    unique_labels, label_counts = np.unique(vesicle_layers, return_counts=True)
    nonzero_unique_labels = unique_labels[unique_labels!=0]
    # print(f"unique_labels are {unique_labels}")
    # print(f"nonzero_unique_labels are {nonzero_unique_labels}")

    for label in nonzero_unique_labels:   
        
        binary_msk = np.copy(vesicle_layers)
        binary_msk[binary_msk != label] = 0
        binary_msk[binary_msk == label] = 1
        
        np.set_printoptions(threshold=np.inf)
        
        # print(f"the binary mask for {label} is {binary_msk}")

### Step 2: connected-component labeling ###
    # obtain a labeled array s.t. all the connected regions have the same integer value
    
        cc_labels = skimage.measure.label(binary_msk, connectivity=1)
        
        unique_cc_labels = np.unique(cc_labels)

        nonzero_labels = unique_cc_labels[np.where(unique_cc_labels!=0)]
        
        if not(np.all(nonzero_labels == 1)):
           
            return False
            
    return True

if __name__ == "__main__": 
    
    chunk_label = 'c2_15'
    
    for synapse_number in range(10):
        
        ds_path = f'/groups/funke/home/luk/synister/run/data/20210525.zarr/synapses_{chunk_label}/{synapse_number}'
    
        zarr_file = zarr.open(ds_path, 'r')
        
        # zarr_file = zarr.open('~/synister/run/data/20210525.zarr/synapses_c0_0/0', 'r')
        # print(f"zarr file keys{zarr_file.tree()}, zarr file info{zarr_file.info}")

        vesicle_layers = zarr_file['vesicles'][:]
        
        print(f"IS synapses_{chunk_label}/{synapse_number} UNIQUE [Expect True]? {check_unique_vesicles(vesicle_layers)}")
    

    ### CHECK ###
    # artificial_vesicle_layer = fake_nonunique_vesicles(annotated_vesicle_layer)
    # print(f"is artificial_vesicle_layer unique [Expect False]? {check_unique_vesicles(artificial_vesicle_layer)}")
    # NOTICE: annotated_layer has been modified HERE


