#Segmentation_functions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
import nibabel as nib

from scipy.ndimage import binary_dilation
from sklearn.model_selection import train_test_split

#### Pre-process Funciton ####
def load_data(path, resize = False,mask = False ,new_x = 256, new_y = 256 ):
    data = nib.load(path).get_fdata()
    
    if resize == True:
        if mask == True:
            data = tf.image.resize(data, [new_x, new_y], method = "nearest")
        if mask == False:
            data = tf.image.resize(data, [new_x, new_y], method = "bicubic")
    return data

def mask_data(mask, data, dx = 5, dy = 5, dz = 1):
    # Create a binary mask where slices_wmgm is 1 or 2
    binary_mask = np.logical_or(mask == 1, mask == 2)

    # Expand the border of the mask by a different number of pixels in x, y, and z directions
    expanded_mask = binary_dilation(binary_mask, structure=np.ones((dx, dy, dz)))

    # Set values in Simon_struc_all to 0 where the expanded mask is False
    data_masked = data.numpy().copy()
    data_masked[~expanded_mask] = 0
    return(data_masked)


def normalize(data):
    try:
        norm_data = data / data.max()
    except:
        norm_data = data / data.numpy().max()
    return norm_data


def mean_std_standardize(data):
    """
    Standardizes the input data by subtracting the mean and dividing by the standard deviation.
    
    Args:
        data (numpy.ndarray): Input data (e.g., an image slice or a 2D array).
        
    Returns:
        standardized_data (numpy.ndarray): The standardized data.
    """
    # Calculate mean and std of the data
    mean = np.mean(data)
    std = np.std(data)
    
    # Avoid division by zero by adding a small epsilon value
    epsilon = 1e-8
    standardized_data = (data - mean) / (std + epsilon)
    
    return standardized_data






def extract_slices(mri_volume, mask = False):
    slices = []
    for i in range(mri_volume.shape[2]):  # Iterate over z-axis
        slice_img = mri_volume[:, :, i]  # Extract single slice (HxW)
        if mask == False:
            slice_img = mean_std_standardize(slice_img) #slice_img / slice_img.numpy().max()  # Normalize

        slices.append(np.expand_dims(slice_img, axis=-1))  # Add channel dim
    return np.array(slices)  # Shape: (1919, 256, 256, 1)


def augment(input_image, input_mask):
   if tf.random.uniform(()) > 0.5:
       # Random flipping of the image and mask
       input_image = tf.image.flip_left_right(input_image)
       input_mask = tf.image.flip_left_right(input_mask)

   return input_image, input_mask



### Compute Sample weights ###
from sklearn.utils.class_weight import compute_class_weight


def sample_weight(y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train),  y = y_train.flatten())

    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))


    ### Sample weights ###
    # Initialize sample weight array (same shape as y_train)
    sample_weights = np.zeros(y_train.shape)  # Shape: (num_samples, height, width)

    # Assign class weights to each pixel
    for class_value, weight in class_weight_dict.items():
        sample_weights[y_train == class_value] = weight

    # Expand dimensions to match one-hot encoded shape
    #sample_weights = np.expand_dims(sample_weights, axis=-1)  # Shape: (num_samples, height, width, 1)

    print(f"shape of sample weights: {sample_weights.shape}")

    return(sample_weights)


