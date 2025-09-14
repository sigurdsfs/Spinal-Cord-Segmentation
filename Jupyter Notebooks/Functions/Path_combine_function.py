import os
from pathlib import Path
from time import time
from dipy.io.image import load_nifti, save_nifti
import numpy as np
import ants

def PathFinder(searchword_starts_with):

    ####### Find the path to all RARE Files ###########
    # Specify the root directory where the search starts
    root_dir = "C:/DTI_SC/humanSC_400h"

    # List to store the paths of files starting with "RARE"
    rare_file_paths = []

    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current folder ends with "HumanSC"
        if dirpath.endswith("HumanSC"):
            # Iterate through each subdirectory within the "HumanSC" folder
            for subdir in dirnames:
                # Create the full path to the subdirectory
                subdir_path = os.path.join(dirpath, subdir)
                
                # Look for files starting with "RARE" in this subdirectory
                for filename in os.listdir(subdir_path):
                    if filename.startswith(searchword_starts_with):
                        # Append the full path of the file to the list
                        rare_file_paths.append(os.path.join(subdir_path, filename))

    # Print or use the paths of the files found
    print(rare_file_paths)
    print("Found N files: ", rare_file_paths.__len__())
    print("Found Paths Done")
    return(rare_file_paths)


def Combine_NIFTI(file_paths, crop_z = True):
    ######### Combine all the RARE Files #######
    # Dictionary to store the data and affine matrices for each file
    nifti_list = None

    # Loop through each file path and load data, affine for each file
    for idx, file_path in enumerate(file_paths):
        #find the correct paths to file and outcome directory.
        file_path = Path(file_path)
        # Get the parent dictonairy where our file is located
        outcomepath = file_path.parent
        # Get the last part of the path
        last_part = outcomepath.name

        try:
            data, affine = load_nifti(file_path)
        except: 
            data = np.load(file_path)


        if crop_z == True:
            if last_part == "173-nii":
                data = data[:,:,65:80]
                print("edge")
            if last_part == "172-nii":
                data = data[:,:,65:80]
                print("edge")
            

            if file_path == "C:/DTI_SC/humanSC_400h/\190819-HumanSC\\173-nii\\RARE_2D_Ax.nii":
                data = data[:,:,65:80]
                print("edge")
            if file_path == "C:/DTI_SC/humanSC_400h\\190819-HumanSC\\172-nii\\Dti_SE.nii":
                data = data[:,:,65:80]
                print("edge")
            else:
                data = data[:,:,10:80]
        else:
            print(f"Starting idx: {idx}")
            if idx == 1: 
                data = data[:,:,:]
            else:
                data = data[:,:,10:80]

        # If nifti_list is None (first iteration), initialize it with the first data array
        if nifti_list is None:
            nifti_list = data
        else:
            # Concatenate data along axis=2
            nifti_list = np.concatenate((data, nifti_list), axis=2)



    print("Finish Combining Nifti Files")
    print("Output Shape = ", nifti_list.shape)
    
    if "affine" in locals():
        print("Affine exist returning it")
        return(nifti_list, affine)
    else:
        print("No affine, returning combined data")
        return(nifti_list)



def Combine_NIFTI_blend(file_paths = None, overlap_slices = 10, blend = "cosine"):
    for i in range(1,file_paths.__len__()):
        #find the correct paths to file and outcome directory.
        file_path = Path(file_paths[i-1])
        file_path2 = Path(file_paths[i])
        # Get the parent dictonairy where our file is located
        outcomepath = file_path.parent
        # Get the last part of the path
        last_part = outcomepath.name

        if i == 1: 
            section_A = ants.image_read(file_paths[i-1])
            section_B = ants.image_read(file_paths[i])
        else: 
            section_A = stitched_image
            section_B = ants.image_read(file_paths[i])


        # Prepare arrays
        try: 
            A_array = section_A.numpy()
        except:
            A_array = section_A

        B_array = section_B.numpy()

        if section_A.dimension == 3 & section_B.dimension == 3:
            if blend == 'linear':
                # Weighted blend in the overlap
                weight_A = np.linspace(0, 1, overlap_slices).reshape(1, 1, -1)
                weight_B = 1 - weight_A

            if blend == 'cosine':
                # Cosine blend from 0 to 1 over overlap_slices
                z = np.linspace(0, np.pi, overlap_slices).reshape(1, 1, -1)
                weight_A = (1 - np.cos(z)) / 2 # 0 at the start, 1 at the end following a cosine curve 
                weight_B = 1 - weight_A  # Inverse of weight_A so it sums to 1. 


            A_overlap = A_array[:, :, :overlap_slices] * weight_A
            B_overlap = B_array[:, :, -overlap_slices:] * weight_B
            blended_overlap = A_overlap + B_overlap

            # Concatenate: A (non-overlap) + blended + B (non-overlap)
            stitched_data = np.concatenate([
                B_array[:, :, :-overlap_slices] , blended_overlap , A_array[:, :, overlap_slices:]
            ], axis=2)

            stitched_image = ants.from_numpy(stitched_data, origin=section_A.origin,
                                                spacing=section_A.spacing, direction=section_A.direction)
        else: 
            if blend == 'linear':
                # Weighted blend in the overlap
                weight_A = np.linspace(0, 1, overlap_slices).reshape(1, 1, -1,1)
                weight_B = 1 - weight_A

            if blend == 'cosine':
                # Cosine blend from 0 to 1 over overlap_slices
                z = np.linspace(0, np.pi, overlap_slices).reshape(1, 1, -1,1)
                weight_A = (1 - np.cos(z)) / 2 # 0 at the start, 1 at the end following a cosine curve 
                weight_B = 1 - weight_A  # Inverse of weight_A so it sums to 1. 


            A_overlap = A_array[:, :, :overlap_slices,:] * weight_A
            B_overlap = B_array[:, :, -overlap_slices:,:] * weight_B
            blended_overlap = A_overlap + B_overlap

            # Concatenate: A (non-overlap) + blended + B (non-overlap)
            stitched_data = np.concatenate([
                B_array[:, :, :-overlap_slices,:] , blended_overlap , A_array[:, :, overlap_slices:,:]
            ], axis=2)

            stitched_image = ants.from_numpy(stitched_data, origin=section_A.origin,
                                                spacing=section_A.spacing, direction=section_A.direction)            

    return(stitched_image)
