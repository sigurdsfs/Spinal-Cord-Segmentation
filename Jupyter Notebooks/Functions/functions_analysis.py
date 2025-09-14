from functions.functions_analysis import *
from functions.Path_combine_function import *
from functions.plot_functions import *
from functions.Preproc_functions import *

#from Path_combine_function import *
#from Preproc_functions import * 
#from plot_functions import *
import matplotlib.pyplot as plt 
import time
import dipy.reconst.dti as dti

import pickle #loading pkl files (dictonairy)



import IPython.display


def diffusion_tensor_model(start_search_word = None, output_name = None, save = True):

    out_path = os.path.join('C:/DTI_SC/humanSC_400h/data', output_name) + ".tkl"
    
    if os.path.exists(out_path):
        print("File Exist already! Loading it in!")

        with open(out_path, 'rb') as f:
            TensorModelDic_final = pickle.load(f)

        return(TensorModelDic_final)
    else:
        print("Starting Analysis")
        ### Diffuison Tensor model
        # Find the different paths to files
        paths_to_data = PathFinder(searchword_starts_with = start_search_word)
        fbval = 'C:/DTI_SC/humanSC_400h/bval.txt'
        fbvec = 'C:/DTI_SC/humanSC_400h/MPG80_bvec.txt'
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)

        TensorModelDic = {}
        for file_path in paths_to_data:
            t = time.time()
            root_dir = "C:/DTI_SC/humanSC_400h"
            #find the correct paths to file and outcome directory.
            file_path = Path(file_path)
            # Get the parent dictonairy where our file is located
            outcomepath = file_path.parent
            # Get the last part of the path
            last_part = outcomepath.name

            #load data
            print(outcomepath, "---- Starting")
            data, affine = load_nifti(file_path)
            
            
            plt.imshow(data[:,:,10,0])
            plt.title("check axis!!")
            plt.show()

            #Calculate Mask based on fully pre-processed data
            maskdata, mask = median_otsu(data, vol_idx= np.arange(0,80), median_radius=4, numpass=4, autocrop=False, dilate=2)
            
            #Fit the TensorModel 
            tenmodel = dti.TensorModel(gtab)
            tenfit = tenmodel.fit(maskdata)

            #Save in dict
            TensorModelDic[last_part] = tenfit

            plt.close()
            IPython.display.clear_output(wait = False)
           
        if save == True: 
            # Save the raw dictionary with TensorFit objects
            out_path = os.path.join('C:/DTI_SC/humanSC_400h/data', output_name) + ".tkl"
            with open(out_path, 'wb') as f:
                pickle.dump(TensorModelDic, f)
        print("Done!")
        return(TensorModelDic)
    



