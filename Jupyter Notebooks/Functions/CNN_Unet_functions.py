#### CNN U-net Functions ####
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

###### Model Architecture ####
def double_conv_block(x, n_filters):

   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

   return x


def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)

   return f, p



def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)

   return x

def build_unet_model():
 # inputs
   inputs = layers.Input(shape=(256,256,1))

   # Encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)

   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)

   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)

   # outputs
   outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

   return unet_model

def build_unet_model_binary():
 # inputs
   inputs = layers.Input(shape=(256,256,1))

   # Encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)

   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)

   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)

   # outputs
   outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

   return unet_model



#### Loss-Functions ####
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Compute Dice Loss.
    
    Args:
        y_true: Ground truth labels (binary or one-hot encoded).
        y_pred: Model predictions (logits or probabilities).
        smooth: Small constant to avoid division by zero.

    Returns:
        Dice Loss value.
    """
    y_true_f = K.flatten(y_true)  # Flatten tensors
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    return 1 - dice_coeff  # Dice Loss = 1 - Dice Coefficient

def masked_categorical_crossentropy(y_true, y_pred):
    """
    Custom loss function that ignores background (class 0).
    """
    loss_fn = CategoricalCrossentropy(from_logits=False)  # Use from_logits=True if no softmax in last layer
    
    # Create mask: 1 for GM/WM, 0 for background
    mask = tf.reduce_max(y_true, axis=-1)  # Get max along last axis (to check if non-zero class is present)
    
    # Compute normal categorical cross-entropy loss
    loss = loss_fn(y_true, y_pred)
    
    # Apply the mask: ignore loss contribution from background voxels
    loss *= mask
    
    # Average over non-background voxels
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_binary_crossentropy(y_true, y_pred):
    """
    Custom loss function that ignores background (class 0) in binary segmentation.
    
    Args:
        y_true: Ground truth labels (batch, H, W, 1)
        y_pred: Predicted probabilities (batch, H, W, 1)
    
    Returns:
        Masked binary cross-entropy loss
    """
    # Define binary cross-entropy loss
    bce = BinaryCrossentropy(from_logits=False)

    # Create mask: 1 for WM/GM (positive class), 0 for background
    mask = tf.cast(y_true > 0, tf.float32)  # Ensures background is ignored

    # Compute binary cross-entropy
    loss = bce(y_true, y_pred)

    # Apply the mask to ignore background
    masked_loss = loss * mask

    # Normalize by number of foreground pixels to prevent bias
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

import matplotlib.pyplot as plt
def plot_history(historybin):
   # Plot training & validation loss values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(historybin.history['loss'], linestyle=':', marker='o', alpha=0.5)
    plt.plot(historybin.history['val_loss'], linestyle=':', marker='o', alpha=0.5)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation AUC values
    plt.subplot(1, 3, 2)
    plt.plot(historybin.history['auc'], linestyle=':', marker='o', alpha=0.5)
    plt.plot(historybin.history['val_auc'], linestyle=':', marker='o', alpha=0.5)
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation accuracy values
    plt.subplot(1, 3, 3)
    plt.plot(historybin.history['accuracy'], linestyle=':', marker='o', alpha=0.5)
    plt.plot(historybin.history['val_accuracy'], linestyle=':', marker='o', alpha=0.5)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot ROC curve
    plt.figure(figsize=(12, 4))

    # True Positives and False Positives
    plt.subplot(1, 2, 1)
    plt.plot(historybin.history['tp'], linestyle=':', marker='o', alpha=0.5, label='True Positives')
    plt.plot(historybin.history['fp'], linestyle=':', marker='o', alpha=0.5, label='False Positives')
    plt.title('True Positives and False Positives')
    plt.ylabel('Count')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # True Negatives and False Negatives
    plt.subplot(1, 2, 2)
    plt.plot(historybin.history['tn'], linestyle=':', marker='o', alpha=0.5, label='True Negatives')
    plt.plot(historybin.history['fn'], linestyle=':', marker='o', alpha=0.5, label='False Negatives')
    plt.title('True Negatives and False Negatives')
    plt.ylabel('Count')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()
   