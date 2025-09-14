import os
#DIPY Plot
from dipy.viz import window, actor
from dipy.data import get_sphere
import matplotlib.pyplot as plt
import time

import keyboard  # For detecting keypresses
import IPython
import numpy as np

def plot_color_fa(tenfit, ax_slice, save = False, save_path = None, plot = True):
    if save_path == None:
        save_path = os.path.join(os.getcwd(),"_fig")

    #Plot 
    dti_fit = tenfit
    sphere = get_sphere('repulsion724')


    scene = window.Scene()

    evals = dti_fit.evals[:, :, ax_slice:ax_slice + 1]
    evecs = dti_fit.evecs[:, :, ax_slice:ax_slice + 1]

    cfa = dti_fit.color_fa[:, :, ax_slice:ax_slice + 1]
    cfa /= cfa.max()

    scene.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere,
                                scale=0.5))


    if save == True:
        print('Saving illustration as tensor_ellipsoids.png')
        window.record(scene, n_frames=1, out_path=save_path,
                    size=(1640 , 1640 ))

    if plot == True :
        window.show(scene)
    return(window)





def plot_all_slices(data, pause = 0.2, increment = 1):
    z_size = data.shape[2]
    for z in range(0,z_size, increment):
        plt.imshow(data[:,:,z])
        plt.title(f"Slice: {z}")
        plt.show()
        time.sleep(pause)  # Small delay to avoid multiple triggers
        IPython.display.clear_output(wait=True)
        plt.close()
    
    
## ODF Plotting
import keyboard  # For detecting keypresses
import IPython
import numpy as np

def plot_ODF(data = None, csaodfs = None,mask = None, interactive = False, save = False, save_path_folder = None,z_increment = 20):

    scene = window.Scene()

    if mask != None:
        ## Masking of the Data and ODF
        mask_expanded = np.expand_dims(mask, axis=-1)  # Shape becom s (96, 96, 187, 1)
        # Apply the mask to data_subset
        data[~mask] = 0

    b0_image = data[:,:,:,0]

    #Rotate Data
    sphere = get_sphere('repulsion724')

    for z_slice in range(b0_image.shape[2]):
        #B0 Actor
        b0_actor = actor.slicer(b0_image) 
        b0_actor.display(z=z_slice)

        #ODF Actor 
        csa_odfs_actor = actor.odf_slicer(
            csaodfs, sphere=sphere, colormap="plasma", scale=0.4)
        csa_odfs_actor.display(z=z_slice)

        #Text
        title_text = actor.text_3d(f"Z-slice =  {z_slice*z_increment} ", position = (10,10,(z_slice+10)), font_size= 8)

        #Add to Scene
        scene.add(title_text)
        scene.add(csa_odfs_actor)
        scene.add(b0_actor)


        
        if os.path.isdir(save_path_folder):
            print("Folder exists")
        else:
            os.mkdir(save_path_folder)


        if save == True:
            print(f"Saving illustration as zslice_{z_slice*z_increment}.png")
            output_name = os.path.join(save_path_folder, f"zslice_{z_slice*z_increment}.png")
            window.record(scene=scene, n_frames=1, out_path=output_name, size=(3080, 3080), magnification= 2, screen_clip= True)
        

        #Show it 
        if interactive == True:
            window.show(scene)
            while True:
                if keyboard.is_pressed('space'):        
                    print("Continuing...") 
                    #plt.close('all')  # Close all matplotlib plots
                    IPython.display.clear_output(wait=True)
                    time.sleep(0.05)  # Small delay to avoid multiple triggers
                    break

        scene.clear()