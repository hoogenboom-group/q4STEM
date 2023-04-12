import os
import tqdm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from q4stem import simul


def details(directory, sett, x, y):
    """ 
    -------------------------------------------------------------------------------
    To show step-by-step individual point preprocesing
    -------------------------------------------------------------------------------

    Inputs: 
        directory ... dataset directory
        sett      ... experiment settings
        x         ... chosen datapoint row
        y         ... chosen datapoint column 
    Output:
        radprof ... azimuthally integrated intensities - radial profile
    """
    
    filename = str(x)+'_'+str(y)+'.'+sett.result[1].value
    I = read(filename, directory)   
    radprof = prep(I, sett, True);  
    return(radprof)

def read(filename, directory):  
    """ 
    -------------------------------------------------------------------------------
    Individual datafile reading
    -------------------------------------------------------------------------------

    Inputs: 
        directory ... dataset directory
        filename  ... name of a file
    Output:
        I         ... scattering/diffraction pattern image
    """
    
    if filename[filename.find('.'):filename.find('.')+4] in ['.tiff', '.tif']: # because of "Thumbs.db" file
            I = np.double(plt.imread(os.path.join(directory, filename)))     
            
    elif filename[filename.find('.'):filename.find('.')+4] in ['.dat']: # for .dat files
            I = np.fromfile(os.path.join(directory, filename), dtype=np.uint16) 
            I = np.double(I.reshape(max_res,max_res)) 
    return(I)


def mapping(directory, sett):
    """ 
    -------------------------------------------------------------------------------
    Mapping the thickness calculation across whole dataset
    -------------------------------------------------------------------------------

    Inputs: 
        directory ... dataset directory
        sett      ... experiment settings
    Outputs:
        Pmap      ... thickness map created with "mcsa" method
        Dmap      ... thickness map created with "dfbf" method
    """
     
    sample  = sett.result[0].value
    field   = sett.result[2].value
    obj     = sett.result[3].value
    depth   = sett.result[4].value
    pix     = sett.result[5].value
    max_res = sett.result[6].value
    binning = sett.result[7].value
    
    factor = np.arctan(binning*(pix/obj)/depth)*1000 # between pixels and angle in mrad"
    Pmap = np.zeros((field,field)) # initialization of Peak Maximum Position based thickness map
    Dmap = np.zeros((field,field)) # initialization of Dark field / Bright field ration based thickness map

    for filename in tqdm.tqdm(os.listdir(directory)):
        if filename[filename.find('.'):filename.find('.')+4] in ['.tiff', '.tif', '.dat']: # because of "Thumbs.db" file
            I = read(filename, directory)  
            x = filename[0:filename.find('_')]
            y = filename[filename.find('_')+1:filename.find('.')]
            radprof = prep(I, sett, 'False')      # data preparation function
            Pmap[int(x)-1,int(y)] = mcsa(radprof, sett, 'False')
            Dmap[int(x)-1,int(y)] = dfb(radprof, sett, 'False') 

    return([Pmap, Dmap])

def mapshow(maps):    
    """ 
    -------------------------------------------------------------------------------
    To show resulting thickness maps
    -------------------------------------------------------------------------------

    Inputs: 
        maps ... computed thickness maps
    """
    
    high = int(np.max(np.array([np.max(maps[0]),np.max(maps[1])])))    # upper visualisation limit

    plt.rcParams['figure.figsize'] = [15, 15]
    plt.subplot(1, 2, 1)    
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(maps[0], cmap = 'jet', vmin = 0, vmax = high)
    plt.title('PeakPos')
    cbar=plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Thickness [nm]', rotation=270)

    plt.subplot(1, 2, 2)  
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(maps[1], cmap = 'jet', vmin = 0, vmax = high)
    plt.title('DF/BF ratio')
    cbar=plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Thickness [nm]', rotation=270)

    return

def prep(I, sett, show):    
    """
    -------------------------------------------------------------------------------
    Individual scattering pattern preprocessing
    -------------------------------------------------------------------------------
    
    Inputs:
        I    ... scattering/diffraction pattern image
        sett ... experiment settings
        show ... boolean operator: True - show figures, False - hide figures 
    Output:
        radprof ... azimuthally integrated intensities - radial profile
    """
    
    df_out = 100 # mrad
    obj     = sett.result[3].value
    depth   = sett.result[4].value
    pix     = sett.result[5].value
    max_res = sett.result[6].value
    binning = sett.result[7].value
    
    factor = np.arctan(binning*(pix/obj)/depth)*1000 # between pixels and angle in mrad"

    if np.max(I)>0: # do not use files with no signal ...
               
        # Localisation of image maximum - center
        I = scipy.signal.medfilt2d(I,3) # light median filter to avoid impulse noise
        result = np.where(I == np.amax(I)) # localisation of image maximum
        cog=[np.around(np.mean(result[0])), np.around(np.mean(result[1]))]
        
        # Picking the central area
        size = int(np.around(min(cog)))
        size2 = np.array(cog)
        size2 = [max_res/binning-size2[0], max_res/binning-size2[1]]
        size2 = int(np.around(min(size2)))
        if size > size2:
            size = size2
        else:
            size = size
            
        if size > np.round(df_out/factor):
            cut = I[int(np.around(cog[0]))-size:int(np.around(cog[0]))+size, int(np.around(cog[1]))-size:int(np.around(cog[1]))+size]
            (width,height) = cut.shape
            
            # Masking the chosen area wih distance higher than diameter - picking the pixels in the corners
            x = np.linspace(0, width-1, width)    
            y = np.linspace(0, width-1, width)
            x, y = np.meshgrid(x, y)
            mask = np.sqrt((x-width/2)**2+(y-width/2)**2)
            mask[mask < ((width/2)*1.2)] = 0
            mask[mask > 1] = 1
            mask[mask == 0] = np.nan
            
            # Mean background intensity calculation for tiff files
            bcg = mask*cut
            poz = np.nanmean(bcg)
             #    poz = np.nanmean(np.where(bcg!=0,bcg,np.nan),1)

            #Background subtraction for tiff files
            cut = np.subtract(cut,poz)

            # Center refinement
            cut_cen = scipy.signal.medfilt2d(cut,11) # 11 for tif, 1 for dat
            result = np.where(cut_cen == np.amax(cut_cen)) # localisation of image maximum
            cog=np.array([np.around(np.mean(result[0])), np.around(np.mean(result[1]))])
            size_new = size * 0.9                  # maximal expected shift of the centre is 10% 
            a = int(np.around(cog[0])-size_new)
            b = int(np.around(cog[0])+size_new)
            c = int(np.around(cog[1])-size_new)
            d = int(np.around(cog[1])+size_new)
            final_cut = cut[a:b, c:d]
    
            # Radial sum calculation
            (width,height) = final_cut.shape 
            [X,Y] = np.meshgrid(np.arange(width)-width/2, np.arange(height)-width/2)
            R = np.sqrt(np.square(X) + np.square(Y))
            
            #if R.size==0:
            #    print('Tady je chyba')
            #else: 
            # Initialize variables
            radial_distance = np.arange(1,np.max(R),1)
            radprof       = np.zeros(len(radial_distance))
            index           = 0
            bin_size        = 2

            # Calcualte radial profile
            for i in radial_distance:
                mask2 = np.greater(R, i - bin_size/2) & np.less(R, i + bin_size/2)
                values = final_cut[mask2]
                radprof[index] = np.sum(values)
                index += 1 
                  
            # Visualisation  
            if show == True:
                
                plt.rcParams['figure.figsize'] = [15, 15]
                plt.subplot(1, 5, 1)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.imshow(I,cmap='gray')
                plt.scatter(cog[1], cog[0],color='r')
                plt.title('Maximum ' + str(cog))
                plt.yticks(color='w') 
                plt.xticks(color='w')
                
                plt.subplot(1, 5, 2)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.imshow(cut,cmap='gray')
                plt.title('Image cutout')

                plt.subplot(1, 5, 3)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.imshow(mask*1,cmap='RdYlGn',vmin = 0, vmax = 1)
                plt.title('Background mask')

                plt.subplot(1, 5, 4)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.imshow(cut,cmap='gray',vmin = 0, vmax = 100)
                plt.scatter(width/2, width/2,color='r')
                plt.scatter(cog[1], cog[0], color='b')
                plt.title('After subtraction')

                plt.subplot(1, 5, 5)
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                plt.imshow(final_cut,cmap='gray',vmin = 0, vmax = 100)
                plt.scatter(len(final_cut)/2, len(final_cut)/2,color='b')
                plt.title('After center refinement')      
        else:
            print('Error: field of view is lower than', df_out, 'mrad.')
    else:
        print('There is fully 0 image in the dataset.')
        
    return(radprof)


def mcsa(radprof, sett, show):  
    """
    -------------------------------------------------------------------------------
    Core of " the Most Common Scattering Angle" method
    -------------------------------------------------------------------------------
    
    Inputs:
        radprof  ... azimuthally integrated intensities - radial profile
        sett     ... experiment settings
        show     ... boolean operator: True - show figures, False - hide figures
    Output:
        thi      ... local thickness in nm
    """
    
    sample  = sett.result[0].value
    field   = sett.result[2].value
    obj     = sett.result[3].value
    depth   = sett.result[4].value
    pix     = sett.result[5].value
    max_res = sett.result[6].value
    binning = sett.result[7].value
    factor = np.arctan(binning*(pix/obj)/depth)*1000 # between pixels and angle in mrad"
    
    # Data upsampling + fitting
    xdata = np.linspace(0,len(radprof)-1, len(radprof))
    xxdata = np.linspace(0,len(radprof)-1, 10*len(radprof)-9)
    p = np.polyfit(xdata, radprof, 9)
    fit_y = np.polyval(p,xxdata)
        
    # Maximum detection in pixels
    peak, _ = scipy.signal.find_peaks(fit_y, height=max(fit_y)) # only the highest peak is detected
    peak2 = peak/10
   
    # Control for empty maximum peak position
    if peak.size == 0:
        peak2 = 0
            
    # Peak position from pixels to mrad
    pos = peak2*factor  
    
    # Calling simulation results
    sim = simul.load(sample, 'mcsa')  
    
    # From mrad to thickness - prapared for up to fourth order polynom fit of simulation data
    thi = sim[0]*np.power(pos,4) + sim[1]*np.power(pos,3) + sim[2]*np.power(pos,2) + sim[3]*pos + sim[4]
    
    # Visualisation 
    if show == True:
        ylim_low = int(np.round(np.min(radprof)-0.05*(np.max(radprof)-np.min(radprof))))
        ylim_high = int(np.round(np.max(radprof)+0.05*(np.max(radprof)-np.min(radprof))))
        
        plt.rcParams['figure.figsize'] = [16, 5]
        plt.subplot(1, 2, 1)
        plt.plot(xdata*factor,radprof,label="radial profile",linewidth=3.0)
        plt.plot(xxdata*factor,fit_y, label="polynomial fit",linewidth=3.0)      
        plt.plot((peak2*factor),(fit_y[peak]),marker=7,markersize=15,label="MCSA",color="red")
        plt.legend(loc="upper right",fontsize = 12)
        plt.title("Azimuthally integrated intensities: Peak maximum",fontsize = 12)
        plt.xlabel("Scattering angle [mrad]",fontsize = 12)
        plt.ylabel("Counts",fontsize = 12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0, 250)       
        plt.ylim(ylim_low, ylim_high)
        plt.text(145, np.max(radprof)/2, ''.join(['MCSA at ',str(np.round(np.float(peak2),2)), ' mrad']), fontsize = 12, weight="bold")
        plt.text(145, np.max(radprof)/2.5, ''.join(['Thickness ',str(np.round(np.float(thi),1)), ' nm']), fontsize = 12, weight="bold")
    return(thi)


def dfb(radprof, sett, show):   
    """
    -------------------------------------------------------------------------------
    Core of "virtual DF/BF segment ratio" method
    -------------------------------------------------------------------------------
    
    Inputs:
        radprof  ... azimuthally integrated intensities - radial profile
        sett     ... experiment settings
        show     ... boolean operator: True - show figures, False - hide figures
    Output:
        thi      ... local thickness in nm
    """
    
    sample  = sett.result[0].value
    field   = sett.result[2].value
    obj     = sett.result[3].value
    depth   = sett.result[4].value
    pix     = sett.result[5].value
    max_res = sett.result[6].value
    binning = sett.result[7].value
    factor = np.arctan(binning*(pix/obj)/depth)*1000 # between pixels and angle in mrad"
    bf = 50;      # mrad
    df_in = 50;   # mrad
    df_out = 100; # mrad
    
    # Angular range from mrad to pix
    bf_max = int(np.around(bf/factor))-1 # Virtual STEM segment range from mrad to pixels
    df_min = int(np.around(df_in/factor))-1
    df_max = int(np.around(df_out/factor))-1
                
    # Mean BF and DF signal + ratio
    rat = np.mean(radprof[df_min:df_max])/np.mean(radprof[0:bf_max]) #
               
    # Calling simulation results
    sim = simul.load(sample,'dfbf')     
        
    # From mrad to thickness - prapared for up to fourth order polynom fit of simulation data
    thi = sim[0]*np.power(rat,4) + sim[1]*np.power(rat,3) + sim[2]*np.power(rat,2) + sim[3]*rat + sim[4]

    # Visualisation 
    if show == True:
        ylim_low = int(np.round(np.min(radprof)-0.05*(np.max(radprof)-np.min(radprof))))
        ylim_high = int(np.round(np.max(radprof)+0.05*(np.max(radprof)-np.min(radprof))))
        
        xdata = np.linspace(0,len(radprof)-1, len(radprof))      
        plt.rcParams['figure.figsize'] = [16, 5]
        plt.subplot(1, 2, 2)
        plt.plot(xdata*factor,radprof,label="radial profile",linewidth=3.0)
        plt.title("Azimuthally integrated intensities: DF/BF",fontsize = 12)
        plt.xlabel("Scattering angle [mrad]",fontsize = 12)
        plt.ylabel("Counts",fontsize = 12)
        plt.xlim(0, 250)   
        plt.ylim(ylim_low, ylim_high)
        rect=mpatches.Rectangle((0,ylim_low),bf,ylim_high+np.abs(ylim_low), alpha=0.1, facecolor="red")
        plt.gca().add_patch(rect)
        plt.text(3, 0,'Virtual BF',fontsize=12, color="red", weight="bold")
        rect=mpatches.Rectangle((df_in, ylim_low),df_out-df_in,ylim_high+np.abs(ylim_low), alpha=0.1, facecolor="green")
        plt.gca().add_patch(rect)
        plt.text(df_in+3, 0 ,'Virtual DF',fontsize=12, color="green", weight="bold")   
        plt.text(145, np.max(radprof)/2, ''.join(['DF/BF ratio ',str(np.round(rat,2))]), fontsize = 12, weight="bold")
        plt.text(145, np.max(radprof)/2.5, ''.join(['Thickness ',str(np.round(thi,1)), ' nm']), fontsize = 12, weight="bold")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="upper right",fontsize = 12)
            
    return(thi)  