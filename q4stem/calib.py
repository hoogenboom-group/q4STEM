import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from q4stem import q4stem


def mapping(directory, sett):
    """ 
    -------------------------------------------------------------------------------
    Mapping saturation control across whole dataset
    -------------------------------------------------------------------------------
    Inputs: 
        directory ... dataset directory
        sett      ... experiment settings
    """
    
    field   = sett.result[2].value
    
    sat  = np.zeros((field,field))  # initialization of saturation map check
    for filename in tqdm.tqdm(os.listdir(directory),desc="File processing"):
        if filename[filename.find('.'):filename.find('.')+4] in ['.tiff', '.tif','.dat']: # because of "Thumbs.db" file
            I = q4stem.read(filename, directory)      # read the file and convert do double
            level = control(I)                  # calibration function
            x = filename[0:filename.find('_')]                              # point coordinates extraction from file name
            y = filename[filename.find('_')+1:filename.find('.')]
            sat[int(x)-1,int(y)]  = level                # write to map

    ### Visualisation ###
    plt.rcParams['figure.figsize'] = [13, 13]

    plt.subplot(1, 3, 1)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(sat, cmap = 'hot', vmin = 0, vmax = np.amax(sat))
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title('Saturation control')
    if np.amax(sat)>0.95*2**sett.result[8].value:
        plt.text(np.round(sett.result[2].value/8), np.round(sett.result[2].value/2), 'Saturation warning!!', fontsize = 12, weight="bold",color="magenta")


def control(I):
    """
    -------------------------------------------------------------------------------
    Reads maximal value for each pixel
    -------------------------------------------------------------------------------
    Inputs: 
        I    ... scattering/diffraction pattern image
    """

    # Image saturation check
    level = np.amax(I)
    
    return(level)



