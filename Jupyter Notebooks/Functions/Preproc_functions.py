from scipy.io import savemat
from dipy.denoise.localpca import mppca
import os
from dipy.io.image import load_nifti, save_nifti
from functions.Path_combine_function import *
#from Path_combine_function import *
import nibabel as nib
import numpy as np
from dipy.denoise.gibbs import gibbs_removal
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import pandas as pd
import matplotlib.pyplot as plt

def mppca_denoise(file_paths, output_name):
    # Initialize an empty dictionary
    sigma_dict = {}
    output_name = output_name
    for file_path in file_paths:
        t = time()

        #find the correct paths to file and outcome directory.
        file_path = Path(file_path)
        # Get the parent dictonairy where our file is located
        outcomepath = file_path.parent
        # Get the last part of the path
        last_part = outcomepath.name



        output_file = os.path.join(outcomepath, output_name + ".nii")
        if os.path.isfile(output_file):
            print(output_file, "Already exist/Completed")
        else:
            print(output_file, "Didn't exist- Starting")
            #load data
            data, affine = load_nifti(file_path)

            #Motion Correction on non mask data
            denoised_arr, sigma = mppca(data, patch_radius=2, return_sigma= True)        
            print("Time taken for local MP-PCA ", -t + time())
            
            #Saving Files
            sigma_dict[last_part] = sigma
            
            #Save Image
            denoised_arr = nib.Nifti1Image(denoised_arr,   affine = affine)
            print("Saving Files")
            save_nifti(os.path.join(outcomepath , output_name), denoised_arr.get_fdata(),
                    denoised_arr.affine)
            print("Done!")

    # Save dictionary to a .mat file
    savemat("C:/DTI_SC/humanSC_400h/noise_sigma_dict.mat", sigma_dict)

    return (sigma_dict)


def gibbs_correct(file_paths, output_name):
    # Initialize an empty dictionary
    output_name = output_name
    for file_path in file_paths:
        

        #find the correct paths to file and outcome directory.
        file_path = Path(file_path)
        # Get the parent dictonairy where our file is located
        outcomepath = file_path.parent
        # Get the last part of the path
        last_part = outcomepath.name


        output_file = os.path.join(outcomepath, output_name + ".nii")
        if os.path.isfile(output_file):
            print(output_file, "Already exist/Completed")
        else:
                print(outcomepath, "Gibbs Corrected Didn't exist- Starting")
                #load data
                data, affine = load_nifti(file_path)

                #Motion Correction on non mask data
                data_corrected = gibbs_removal(data, slice_axis=2, num_processes=-1)

                #Saving Files
                gibbs_correc = nib.Nifti1Image(data_corrected,   affine = affine)
                save_nifti(os.path.join(outcomepath, output_name), gibbs_correc.get_fdata(),
                    gibbs_correc.affine)
                print("Done!")
    return("All Done")




def SNR(data_path, option):
    """
    data_path: Is the absolute path to the desired image files. \n
    option:
        option = 1: Look for SNR based on the b-vectors with shortest euclidean distance to the axis-aligned vectors 
        option = 2: SNR in all b-vector directions. 
    """
    #Signal to noise

    fbval = 'C:/DTI_SC/humanSC_400h/bval.txt'
    fbvec = 'C:/DTI_SC/humanSC_400h/MPG80_bvec.txt'
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    gtab = gradient_table(bvals, bvecs)
    data, affine = load_nifti(data_path)
    #Calculate Mask
    maskdata, mask = median_otsu(data, vol_idx= np.arange(0,80), median_radius=3, numpass=1, autocrop=False, dilate=2)

    #Signal to noise
    mean_signal = np.mean(data[mask], axis=0)
    mean_signal

    mask_noise = ~mask #true for outside of target
    noise_std = np.std(data[mask_noise, :]) #subset based on mask's three Dimensions (keep last as is)
    noise_std


    ### Exclude null bvecs from the search
    ## The code effectively computes the Euclidean distance between the target vector [1,0,0] [1,0,0]
    ## and every vector in gtab.bvecs, then finds the index of the vector with the 
    ## smallest distance (i.e., the most similar vector).
    idx = np.sum(gtab.bvecs, axis=-1) == 0
    gtab.bvecs[idx] = np.inf
    axis_X = np.argmin(np.sum((gtab.bvecs-np.array([1, 0, 0]))**2, axis=-1))
    axis_Y = np.argmin(np.sum((gtab.bvecs-np.array([0, 1, 0]))**2, axis=-1))
    axis_Z = np.argmin(np.sum((gtab.bvecs-np.array([0, 0, 1]))**2, axis=-1))

    # Initialize a dictionary to store SNR values
    snr_dict = {}
    if option == 1:
        ###  Signal-to-noise ratio calculations ####
        # For directions with shortest euclidiean distance to axis-aligned-vectors. 
        for direction in [0, axis_X, axis_Y, axis_Z]:
            SNR = mean_signal[direction] / noise_std
            if direction == 0:
                direction_name = "b0"
                print("SNR for the b=0 image is :", SNR)
            elif direction == axis_X:
                direction_name = "X"
                print("SNR for direction X", gtab.bvecs[direction], "is :", SNR)
            elif direction == axis_Y:
                direction_name = "Y"
                print("SNR for direction Y", gtab.bvecs[direction], "is :", SNR)
            elif direction == axis_Z:
                direction_name = "Z"
                print("SNR for direction Z", gtab.bvecs[direction], "is :", SNR)

            # Store the SNR value in the dictionary with the direction as the key
            snr_dict[direction_name] = SNR
    elif option == 2:
        #For all directions
        for direction in range(83):
            SNR = mean_signal[direction] / noise_std
            if direction == 0:
                print(f"SNR for the b=0 image is : {SNR}")
            else:
                print(f"SNR for direction {direction} {gtab.bvecs[direction]} is : {SNR}")
            
            # Store the SNR value in the dictionary with the direction as the key
            snr_dict[f"Direction: {direction} = {gtab.bvecs[direction]} "] = SNR


    # Create a pandas DataFrame with one row and columns for each direction
    snr_df = pd.DataFrame([snr_dict])
    return snr_df


def evaluate_bias_field_correction(original_image = None, corrected_image = None, bias_field=None, xi = 180 , yi = 180, zi = 40):
    """
    Evaluate the bias field correction by comparing the original and corrected images.
    This function calculates the mean squared error (MSE) and displays both images.
    """
    original_data = original_image.numpy()
    corrected_data = corrected_image.numpy()

    # Calculate the mean squared error (MSE) between the original and corrected images
    mse = np.mean((original_data - corrected_data) ** 2)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    # Display the original and corrected images in sagittal and axial views

    ###### Display Xi ######
    plt.subplot(1, 3, 1)
    plt.imshow(original_image[:, xi, :], cmap='gray')
    plt.title("Original (Coronal)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(corrected_image[:, xi, :], cmap='gray')
    plt.title("Corrected (Coronal)")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(bias_field[:, xi, :], cmap='gray')
    plt.title("Bias Field (Coronal View)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    ###### Display zi ######
    plt.subplot(1, 3, 1)
    plt.imshow(original_image[:, :, zi], cmap='gray')
    plt.title("Original (axial)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(corrected_image[:, :, zi], cmap='gray')
    plt.title("Corrected (axial)")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(bias_field[:, :, zi], cmap='gray')
    plt.title("Bias Field (Axial View)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    ###### Display yi ######
    plt.subplot(1, 3, 1)
    plt.imshow(original_image[yi, :], cmap='gray')
    plt.title("Original (Sagittal View)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(corrected_image[yi, :], cmap='gray')
    plt.title("Corrected (Sagittal View)")
    plt.axis("off")


    plt.subplot(1,3,3)
    plt.imshow(bias_field[yi,:, :], cmap='gray')
    plt.title("Original (Sagittal View)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()



    #### Compute mean intensity along z (slice axis) ####
    original_data = original_image.numpy()
    corrected_data = corrected_image.numpy()

    # Mean intensity along axes
    # Only calculate mean intensity on non-zero voxels for each slice
    mean_intensity_before = [
        original_data[:, :, z][original_data[:, :, z] > 0].mean() if np.any(original_data[:, :, z] > 0) else 0
        for z in range(original_data.shape[2])
    ]
    mean_intensity_after = [
        corrected_data[:, :, z][corrected_data[:, :, z] > 0].mean() if np.any(corrected_data[:, :, z] > 0) else 0
        for z in range(corrected_data.shape[2])
    ]
    mean_intensity_before_x = [
        original_data[x, :, :][original_data[x, :, :] > 0].mean() if np.any(original_data[x, :, :] > 0) else 0
        for x in range(original_data.shape[0])
    ]
    mean_intensity_after_x = [
        corrected_data[x, :, :][corrected_data[x, :, :] > 0].mean() if np.any(corrected_data[x, :, :] > 0) else 0
        for x in range(corrected_data.shape[0])
    ]
    mean_intensity_before_y = [
        original_data[:, y, :][original_data[:, y, :] > 0].mean() if np.any(original_data[:, y, :] > 0) else 0
        for y in range(original_data.shape[1])
    ]
    mean_intensity_after_y = [
        corrected_data[:, y, :][corrected_data[:, y, :] > 0].mean() if np.any(corrected_data[:, y, :] > 0) else 0
        for y in range(corrected_data.shape[1])
    ]

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(mean_intensity_before, label='Before', linestyle='--')
    plt.plot(mean_intensity_after, label='After', linestyle='-')
    plt.xlabel('Slice (z-axis)')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity (z-axis)')
    plt.legend()
    plt.grid()

    # Add vertical lines every 70th slice
    for x in range(0, len(mean_intensity_before), 70):
        plt.axvline(x=x, color='r', linestyle=':', alpha=0.5)

   

    plt.subplot(1, 3, 2)
    plt.plot(mean_intensity_before_x, label='Before', linestyle='--')
    plt.plot(mean_intensity_after_x, label='After', linestyle='-')
    plt.xlabel('Slice (x-axis)')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity (x-axis)')
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(mean_intensity_before_y, label='Before', linestyle='--')
    plt.plot(mean_intensity_after_y, label='After', linestyle='-')
    plt.xlabel('Slice (y-axis)')
    plt.ylabel('Mean Intensity')
    plt.title('Mean Intensity (y-axis)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # Coefficient of variation (CV) across slices

    cv_before = np.std(mean_intensity_before) / np.mean(mean_intensity_before)
    cv_after = np.std(mean_intensity_after) / np.mean(mean_intensity_after)
    print(f"CV before: {cv_before:.4f}, CV after: {cv_after:.4f}")


    # Histogram of intensities before and after correction
    # Flatten and remove zeros
    original_flat = original_data[original_data > 0].flatten()
    corrected_flat = corrected_data[corrected_data > 0].flatten()


    plt.hist(original_flat, bins=100, alpha=0.5, label='Before')
    plt.hist(corrected_flat, bins=100, alpha=0.5, label='After')
    plt.legend()
    plt.title("Histogram of Intensities")
    plt.show()
    
    return mse