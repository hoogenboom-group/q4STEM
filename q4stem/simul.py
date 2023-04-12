import numpy as np
import matplotlib.pyplot as plt

def load(sample, typ):
    
    """ 
    -------------------------------------------------------------------------------
    Monte Carlo electron scattering simulation results - loading
    -------------------------------------------------------------------------------

    Inputs: 
        sample ... 'ice', 'latex', 'resin' or 'carbon'
        typ    ... 'mcsa' for the most common scattering angle method, 'dfbf' for the relative signal ratio of virtual DF and BF segments method
    """
             
    if typ in ['mcsa']:
        
        coef = np.array([[0, 0, 0, 6.6195, 174.4491],[0, 0, 0, 7.7751, 55.3670],[0, 0, 0, 6.9255, 28.4519],[0, 0, 0, 4.4271, 5.2997]])
        # polynomial fit coeficients: p1*np.power(ratio,4) + p2*np.power(ratio,3) + p3*np.power(ratio,2) + p4*ratio + p5
        
        if sample in ['ice']:
            sim = coef[0,:]
        elif sample in ['latex']:
            sim = coef[1,:]
        elif sample in ['resin']:
            sim = coef[2,:]
        elif sample in ['carbon']:
            sim = coef[3,:]
        else:
            sys.exit("Check the chosen sample.")

    elif typ in ['dfbf']:
        
        coef = np.array([[0, 0, -50.7525, 492.5150 , 4.3625],[0, 0, -17.6362, 377.3019, 7.6086],[0, 0, -4.3966, 309.5517, 7.6832],[0, 0, 32.9870, 129.3026 , 16.9842]])
        # polynomial fit coeficients: p1*np.power(rat,4) + p2*np.power(rat,3) + p3*np.power(rat,2) + p4*rat + p5
        
        if sample in ['ice']:
            sim = coef[0,:]
        elif sample in ['latex']:         
            sim = coef[1,:]
        elif sample in ['resin']:
            sim = coef[2,:]
        elif sample in ['carbon']:
            sim = coef[3,:]
        else:
            sys.exit("Check the chosen sample.")    
    else:
        sys.exit("Check the chosen method.")    
    return(sim)


def show(typ):
    """ 
    -------------------------------------------------------------------------------
    Monte Carlo electron scattering simulation results - visualisation
    -------------------------------------------------------------------------------
    Inputs:
        typ    ... 'mcsa' for the most common scattering angle method, 'dfbf' for the relative signal ratio of virtual DF and BF segments method
    """ 
        
    if typ in ['mcsa']:
        x = np.linspace(0,100-1, 100) # Position from 0 up to 100 mrad
            
        coefi = load('ice', 'mcsa')
        coefl = load('latex', 'mcsa')
        coefe = load('resin', 'mcsa')
        coefc = load('carbon', 'mcsa')
            
        yi = coefi[0]*np.power(x,4) + coefi[1]*np.power(x,3) + coefi[2]*np.power(x,2) + coefi[3]*x + coefi[4]
        yl = coefl[0]*np.power(x,4) + coefl[1]*np.power(x,3) + coefl[2]*np.power(x,2) + coefl[3]*x + coefl[4]
        ye = coefe[0]*np.power(x,4) + coefe[1]*np.power(x,3) + coefe[2]*np.power(x,2) + coefe[3]*x + coefe[4]
        yc = coefc[0]*np.power(x,4) + coefc[1]*np.power(x,3) + coefc[2]*np.power(x,2) + coefc[3]*x + coefc[4]

        plt.rcParams['figure.figsize'] = [15, 5]
        plt.subplot(1, 3, 2)
        plt.plot(x,yi,label="ICE",linewidth=4.0)
        plt.plot(x,yl,label="LATEX",linewidth=4.0)
        plt.plot(x,ye,label="RESIN",linewidth=4.0)
        plt.plot(x,yc,label="CARBON",linewidth=4.0)
        plt.legend(loc="lower right",fontsize = 12)
        plt.title("Peak maximum position",fontsize = 12)
        plt.xlabel("MCSA [mrad]",fontsize = 12)
        plt.ylabel("Thikncess [nm]",fontsize = 12)
        plt.xlim(0, 100)   
        plt.ylim(0, 500)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    elif typ in ['dfbf']:
            
        coefi = load('ice', 'dfbf')
        coefl = load('latex', 'dfbf')
        coefe = load('resin', 'dfbf')
        coefc = load('carbon', 'dfbf')

        x = np.linspace(0,2,num=100) # Ratio from 0 up to 2
        yi = coefi[0]*np.power(x,4) + coefi[1]*np.power(x,3) + coefi[2]*np.power(x,2) + coefi[3]*x + coefi[4]
        yl = coefl[0]*np.power(x,4) + coefl[1]*np.power(x,3) + coefl[2]*np.power(x,2) + coefl[3]*x + coefl[4]
        ye = coefe[0]*np.power(x,4) + coefe[1]*np.power(x,3) + coefe[2]*np.power(x,2) + coefe[3]*x + coefe[4]
        yc = coefc[0]*np.power(x,4) + coefc[1]*np.power(x,3) + coefc[2]*np.power(x,2) + coefc[3]*x + coefc[4]

        plt.rcParams['figure.figsize'] = [15, 5]
        plt.subplot(1,3, 3)
        plt.plot(x,yi,label="ICE",linewidth=4.0)
        plt.plot(x,yl,label="LATEX",linewidth=4.0)
        plt.plot(x,ye,label="RESIN",linewidth=4.0)
        plt.plot(x,yc,label="CARBON",linewidth=4.0)
        plt.legend(loc="lower right",fontsize = 12)
        plt.title("DF/BF segment signal ratio",fontsize = 12)
        plt.xlabel("DF/BF ratio",fontsize = 12)
        plt.ylabel("Thikncess [nm]",fontsize = 12)
        plt.xlim(0, 2)        
        plt.ylim(0, 500)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
    else:
        print('Method is not recogised.')
    return

