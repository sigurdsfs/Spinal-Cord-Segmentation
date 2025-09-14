import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

def dice_loss(pred, target, smooth=.25):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, H, W), 1.
        target: Tensor of ground truth (batch_size, H, W, 1).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    #pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()

class PolyLRScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super(PolyLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power for base_lr in self.base_lrs]

###### Calculate Metrics ###### 
from sklearn.metrics import roc_auc_score

def calculate_metrics(predictions, targets, criterion,device):
    #targets = torch.tensor(targets.squeeze()).to(device)  # Ensure targets are on the same device as predictions
    #predictions = torch.tensor(predictions.squeeze()).to(device)  # Ensure predictions are on the same device as targets    
    targets = targets.squeeze().to(device)
    predictions = predictions.squeeze().to(device)

    metrics = {}

    # Calculate loss
    loss = criterion(predictions, targets).item()
    metrics['loss'] = loss

    # Apply threshold to get binary predictions
    predicted = torch.where(predictions > 0.5, 1, 0)

    # Calculate True Positive, False Positive, True Negative, False Negative
    tp = ((predicted == 1) & (targets == 1)).sum().item()
    fp = ((predicted == 1) & (targets == 0)).sum().item()
    tn = ((predicted == 0) & (targets == 0)).sum().item()
    fn = ((predicted == 0) & (targets == 1)).sum().item()

    metrics['true_positive'] = tp
    metrics['false_positive'] = fp
    metrics['true_negative'] = tn
    metrics['false_negative'] = fn

    # Calculate binary accuracy
    total = targets.numel()
    correct = (predicted == targets).sum().item()
    accuracy = correct / total
    metrics['accuracy'] = accuracy

    # Calculate AUC
    auc = roc_auc_score(targets.cpu().numpy().flatten(), predictions.cpu().detach().numpy().flatten())
    metrics['auc'] = auc

    return metrics


##### Validation Function ##### 
def evaluate_model(weights_file ,test_loader, criterion, device,threshold = 0.5):
    onnx_model_path = "C:/Users/Bruger/sct_6.4/data/deepseg_gm_models/large_model.onnx"
    onnx_model = onnx.load(onnx_model_path)  # Load the ONNX model
    onnx.checker.check_model(onnx_model)  # Check if the model is valid
    pytorch_model = convert(onnx_model)  # Convert the ONNX model to a PyTorch model

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device= device)
    print(f"Running on {device}")

    #Load model Weights
    checkpoint = torch.load(weights_file, map_location=device,  weights_only= False)
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()  # Set the model to evaluation mode


    #Setup Data Structures:
    y_test_prediction = []  # Initialize a list to store predictions
    all_outputs = []  # Initialize a list to store all outputs

    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0

        for batch_idx, (batch_image, batch_labels) in enumerate(test_loader):
            batch_image = batch_image.to(device)  # Move to device
            batch_labels = batch_labels.to(device)  # Move to device

            # Model output
            outputs = pytorch_model(batch_image)
            all_outputs.append(outputs)  # Save the output

            # Apply threshold to get binary predictions
            predicted = torch.where(outputs > threshold, 1, 0)

            # Save predictions
            y_test_prediction.append(predicted.cpu().numpy())


            # Loss
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item()

            total += batch_labels.numel()
            correct += (predicted == batch_labels).sum().item()

            false_positive += ((predicted == 1) & (batch_labels == 0)).sum().item()
            false_negative += ((predicted == 0) & (batch_labels == 1)).sum().item()
            true_positive += ((predicted == 1) & (batch_labels == 1)).sum().item()
            true_negative += ((predicted == 0) & (batch_labels == 0)).sum().item()

        # Metrics
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        print(f"False Positive: {false_positive}")
        print(f"False Negative: {false_negative}")
        print(f"True Positive: {true_positive}")
        print(f"True Negative: {true_negative}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")

    # Convert y_test_prediction to a NumPy array and remove the second dimension
    y_test_prediction = np.squeeze(np.array(y_test_prediction))

    # Convert all_outputs to a tensor
    probablity_outputs = torch.cat(all_outputs, dim=0)

    return y_test_prediction, probablity_outputs, avg_test_loss, accuracy

#Update to further training
from tqdm import tqdm
import os

def train_model(pytorch_model, train_loader, test_loader, criterion, optimizer, scheduler, start_epoch=0, max_epochs=20, device='cpu', save_path_model=None):
    pytorch_model.train()
    pytorch_model = pytorch_model.to(device)
    epoch_metrics = []  # List to store metrics for each epoch

    parent_dir = os.path.dirname(save_path_model)
    if not os.path.exists(parent_dir):
        print(f"Creating directory: {parent_dir}")
        os.makedirs(parent_dir)

    for epoch in range(start_epoch, max_epochs):  # Start from last saved epoch
        pytorch_model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")

        for batch_idx, (data, target) in enumerate(progress_bar):
            batch_images = data.to(device)
            batch_labels = target.to(device)

            # Forward pass
            outputs = pytorch_model(batch_images)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()
            running_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

            # Update tqdm description dynamically
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        
        # Test Set evaluation
        y_train_bin_prediction, outputs_probability_tensor, avg_test_loss, accuracy = evaluate_model( pytorch_model, 
                                                                                                   test_loader = test_loader,
                                                                                                   criterion=criterion, 
                                                                                                   device=device)

        # Calculate and save metrics
        y_test = test_loader.dataset.tensors[1]
        metrics_test = calculate_metrics(outputs_probability_tensor, y_test, criterion = criterion, device= device)
        metrics_test['epoch'] = epoch + 1
        metrics_test['train_loss'] = running_loss / len(train_loader)

         # Append metrics for the current epoch
        epoch_metrics.append(metrics_test)
        print(f"Validation: Epoch [{epoch+1}/{max_epochs}], Metrics: {metrics_test}")
        
        # Add the epoch_metrics as an attribute to pytorch_model
        pytorch_model.validation_history = epoch_metrics

        #Scheduler step
        scheduler.step()
        print(f"Epoch {epoch+1}: Learning Rate = {scheduler.get_last_lr()[0]:.6f}")
        print(f"Epoch [{epoch+1}/{max_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")


        # Save model checkpoint
        output_path = save_path_model + f'{epoch}of{max_epochs}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': pytorch_model.state_dict(),
            'validation_history': pytorch_model.validation_history,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
        }, output_path)
    
    print("Training complete.")
    return pytorch_model


def load_model(pytorch_model, optimizer, scheduler, load_path_model, device="cpu"):
    # Load the saved checkpoint
    checkpoint = torch.load(load_path_model, map_location=device,  weights_only= False)

    # Load model state dictionary
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model = pytorch_model.to(device)
    try:
        pytorch_model.validation_history = checkpoint["validation_history"]
    except:
        print("No validation history found in the checkpoint.")
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state and manually set last_epoch
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scheduler.last_epoch = checkpoint['epoch']  # Ensure it continues from the correct epoch

    last_epoch = checkpoint['epoch']

    print(f"Model loaded successfully from: {load_path_model}")
    print(f"Resuming training from epoch {last_epoch + 1}")

    return pytorch_model, optimizer, scheduler, last_epoch


import IPython
from IPython.display import display
import keyboard
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
from onnx2torch import convert


def predict_and_visualize(data_tensor = None,threshold = 0.5,save = True, save_path = None, device = None, weights_file = None, plot = True, plot_increment = 3):
    '''Predict and visualize the results of the model on the input data tensor.
    Args: Data_tensor: First dimension is the number of slices, second and third are the height and width of the image.
    threshold: Threshold for binary classification. if none is provided 0.5 is used.
    returns: a prediction probabilities and a binary prediction based on give threshold.
    '''
    
    onnx_model_path = "c:/sct_6.4/data/deepseg_gm_models/large_model.onnx"
    onnx_model = onnx.load(onnx_model_path)  # Load the ONNX model
    onnx.checker.check_model(onnx_model)  # Check if the model is valid
    pytorch_model = convert(onnx_model)  # Convert the ONNX model to a PyTorch model

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model = pytorch_model.to(device= device)
    print(f"Running on {device}")

    #Load model Weights
    checkpoint = torch.load(weights_file, map_location=device,  weights_only= False)
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])

    #Prediction
    pytorch_model.eval()
    print("starting new")
    all_label_predictions = torch.zeros((data_tensor.shape[0], data_tensor.shape[1], data_tensor.shape[2], data_tensor.shape[3]), device=device)
    with torch.no_grad():
        for i in range(data_tensor.shape[0]):
            label_predictions = pytorch_model(data_tensor[i,:,:,:].to(device=device).unsqueeze(0))
            all_label_predictions[i] = label_predictions
            
            #if i == 0:
            #    all_label_predictions = label_predictions
            #else:
            #    all_label_predictions = torch.cat((all_label_predictions, label_predictions), dim=0)
            label_predictions = label_predictions.cpu().detach()  # Ensure it's moved to CPU and detached

    # Convert tensors to numpy arrays
    all_label_predictions_binary_np = np.where(all_label_predictions.cpu().numpy() < threshold, 0, 1)


    if save == True:
        # Save the numpy arrays
        np.save(save_path, all_label_predictions_binary_np)


    if plot == True:
        fig, axes = plt.subplots(1, 3)
        for z in np.arange(0,data_tensor.shape[0], plot_increment):
            if keyboard.is_pressed('esc'):
                print("Exiting loop...")
                break

            axes[0].cla()
            axes[1].cla()
            axes[2].cla()

            axes[0].imshow(data_tensor[z, :, :, 0].cpu(), cmap="gray")
            axes[0].set_title(f"RARE Data (Slice {z+1}/{data_tensor.shape[0]})")

            axes[1].imshow(all_label_predictions[z, :, :, 0].cpu(), cmap="gray")
            axes[1].set_title(f"Label Prediction (Slice {z+1}/{data_tensor.shape[0]})")

            axes[2].imshow(data_tensor[z, :, :, 0].cpu(), cmap="gray")
            axes[2].imshow(all_label_predictions[z, :, :, 0].cpu(), alpha = .5)
            axes[2].set_title(f"Label Prediction Overlay (Slice {z+1}/{data_tensor.shape[0]})")

            
            display(fig)
            IPython.display.clear_output(wait=True)
            plt.close(fig)
    return((all_label_predictions_binary_np, all_label_predictions))


### Quanticer Function ### 

from sklearn.metrics import f1_score, matthews_corrcoef, roc_curve

def find_best_f1_threshold(y_true, y_probs):
    thresholds = np.linspace(0, 1, 100)  # Generate thresholds from 0 to 1
    best_threshold, best_f1 = 0, 0
    
    for t in thresholds:
        print(f"Trying Threshold: {t}")
        y_pred = (y_probs >= t).astype(int)

        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    return best_threshold, best_f1


def find_best_mcc_threshold(y_true, y_probs):
    thresholds = np.linspace(0, 1, 100)
    best_threshold, best_mcc = 0, -1
    
    for t in thresholds:
        print(f"Trying Threshold: {t}")
        y_pred = (y_probs >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, t

    return best_threshold



def find_best_youden_threshold(y_true, y_probs):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    
    return best_threshold


def combine_mask(gm_mask_path = None, wm_mask_path = None, gm_mask = None,wm_mask = None, output_path = None):
    if gm_mask is None:
        #Load the GM and WM masks
        gm_mask = np.load(gm_mask_path)
    if wm_mask is None:
        wm_mask = np.load(wm_mask_path)

    # Create a combined mask with the same shape as the input masks
    combined_mask = np.zeros_like(gm_mask)

    # Set GM mask values to 1
    combined_mask[gm_mask == 1] = 1

    # Set WM mask values to 2, overwriting GM values if both are present
    combined_mask[wm_mask == 1] = 2

    combined_mask = np.transpose(combined_mask, (0,1,2,3))
    
    
    if output_path != None:
        # Save the combined mask as a NIfTI file
        # Convert to int32 (or uint8 if the mask contains only 0s and 1s)
        combined_mask = combined_mask.astype(np.int32)  # or np.uint8
        combined_mask_nifti = nib.Nifti1Image(combined_mask, np.eye(4))
        nib.save(combined_mask_nifti, (output_path +".nii.gz"))

        np.save((output_path + ".npy"), combined_mask)
    return combined_mask


# Resize Image
def load_and_resize(path, mask = False):
    vol = nib.load(path).get_fdata()
    vol = torch.tensor(vol, dtype=torch.float32)

    vol = vol.permute(2, 0, 1)  # Change to (X, Y,Z) -> (Z,X,Y)
    vol = vol.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: (1, 1, Z, X, Y)
    if mask == False:
        resized_vol = F.interpolate(vol, size=(vol.shape[2], 200, 200), mode='trilinear', align_corners=False)
    elif mask == True:
        resized_vol = F.interpolate(vol, size=(vol.shape[2], 200, 200), mode='nearest')

    resized_vol = resized_vol.squeeze(0).squeeze(0)
    return resized_vol



import time
def plot_all_slices(data, pause = 0.2, increment = 1):
    z_size = data.shape[2]
    for z in range(0,z_size, increment):
        plt.imshow(data[:,:,z])
        plt.title(f"Slice: {z}")
        plt.show()
        time.sleep(pause)  # Small delay to avoid multiple triggers
        IPython.display.clear_output(wait=True)
        plt.close()