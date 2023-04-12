import os
import numpy as np
import ipywidgets as widgets
from ipyfilechooser import filechooser 

def directory():
    """ 
    -------------------------------------------------------------------------------
    Dataset path finder
    -------------------------------------------------------------------------------
    """
    
    # Create and display a FileChooser widget
    fc = filechooser.FileChooser()
    display(fc)

    # Change defaults and reset the dialog
    fc.default_path = 'C:\\Users\'
    fc.reset()

    # Change hidden files
    fc.show_hidden = False

    # Switch to folder-only mode
    fc.show_only_dirs = True

    # Change the title (use '' to hide)
    fc.title = '<b>Choose dataset folder</b>'

    # Sample callback function
    def change_title(chooser):
        chooser.title = 'The folder consists of ' + str(len(os.listdir(fc.value))) + ' files.'

    # Register callback function
    fc.register_callback(change_title)

    return fc

def init(pic): 
    
    """ 
    -------------------------------------------------------------------------------
    Experimental settings
    -------------------------------------------------------------------------------
    Output:
        sett ... experiment settings consists of:

            sam       = sett.result[0].value ... sample type {ice, latex, resin, carbon}
            dat       = sett.result[1].value ... datafile format {tiff, dat}
            fiel      = sett.result[2].value ... square map size
            objective = sett.result[3].value ... optical objective magnification {40, 50, 60, 100}
            dep       = sett.result[4].value ... calibration distance [um for light conv det; mm for direct det]
            pixel     = sett.result[5].value ... detector pixel size [um]
            res       = sett.result[6].value ... detector maximum resolution {2048x2048, 1024x1024, 512x512, 256x256, 128x128, 64x64}
            binn      = sett.result[7].value ... detector binning {1, 2, 4, 8}
            bit_depth = sett.result[8].value ... bit depth of a image                       
    """
    
    if pic =='Direct detection':    
         # Both
        sam = widgets.Dropdown(options=['ice', 'latex', 'resin', 'carbon'],value='latex',disabled=False,)
        dat = widgets.Dropdown(options=['tiff','dat'],value='dat', disabled=False,)
        fiel = widgets.IntSlider(value=100, min=0, max=500,step=10,disabled=False,continuous_update=True,orientation='horizontal',readout=True,readout_format='d')

        # Direct
        objective = 1
        dep = widgets.BoundedFloatText(value=40, min=0,max=100,step=1, disabled=False)
        pixel = widgets.BoundedFloatText(value=55, min=0,max=100.0,step=1, disabled=False)
        res = widgets.Dropdown(options=[('2048x2048', 2048), ('1024x1024', 1024), ('512x512', 512), ('256x256', 256), ('128x128', 128), ('64x64', 64)],value=256,disabled=False,)
        binn = widgets.Dropdown(options=[1,2,4,8],value=1, disabled=False,)
        bit_depth = widgets.Dropdown(options=[8,12,13.5,14,16,32],value=13.5, disabled=False,)

        both = widgets.Accordion(children=[sam, fiel, dat])
        both.set_title(0, 'Sample type')
        both.set_title(1, 'Map size')
        both.set_title(2, 'File type')

        direct = widgets.Accordion(children=[pixel, dep, res, binn, bit_depth])
        direct.set_title(0, 'Real pixel size [um]')
        direct.set_title(1, 'Calibration distance [mm]')
        direct.set_title(2, 'Maximum resolution')
        direct.set_title(3, 'Binning')
        direct.set_title(4, 'Bit depth')

        nest = widgets.Tab()
        nest.children = [both, direct]
        nest.set_title(0, 'General')
        nest.set_title(1, 'Direct det. related')

    elif pic =='Light conversion':
        
        # Both
        sam = widgets.Dropdown(options=['ice', 'latex', 'resin', 'carbon'],value='latex',disabled=False,)
        dat = widgets.Dropdown(options=['tiff','dat'],value='tiff', disabled=False,)
        fiel = widgets.IntSlider(value=100, min=0, max=500,step=10,disabled=False,continuous_update=True,orientation='horizontal',readout=True,readout_format='d')

        # Light
        objective = widgets.Dropdown(options=[40,50,60,100],value=60, disabled=False)
        dep = widgets.BoundedFloatText(value=300, min=0,max=50000.0,step=10, disabled=False)
        pixel = widgets.BoundedFloatText(value=6.5, min=0,max=50.0,step=0.1, disabled=False)
        res = widgets.Dropdown(options=[('2048x2048', 2048), ('1024x1024', 1024), ('512x512', 512), ('256x256', 256)],value=2048,disabled=False,)
        binn = widgets.Dropdown(options=[1,2,4,8],value=4, disabled=False,)
        bit_depth = widgets.Dropdown(options=[8,12,14,16,32],value=16, disabled=False,)

        both = widgets.Accordion(children=[sam, fiel, dat])
        both.set_title(0, 'Sample type')
        both.set_title(1, 'Map size')
        both.set_title(2, 'File type')
        
        light = widgets.Accordion(children=[objective, dep, pixel, res, binn, bit_depth])
        light.set_title(0, 'Optical objective magnification')
        light.set_title(1, 'Calibration distance [um]')
        light.set_title(2, 'Real pixel size [um]')
        light.set_title(3, 'Maximum resolution')
        light.set_title(4, 'Binning')
        light.set_title(5, 'Bith depth')

        nest = widgets.Tab()
        nest.children = [both, light]
        nest.set_title(0, 'General')
        nest.set_title(1, 'Light conv. related') 

    display(nest) 
    
    return (sam, dat, fiel, objective, dep, pixel, res, binn, bit_depth)
  #          0    1    2        3       4     5     6     7       8    
