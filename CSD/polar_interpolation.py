# -*- coding: utf-8 -*-

import numpy as np
from coord_conv import cart2pol, pol2cart
from scipy.interpolate import interp1d

def polar_interpolation(curve, c_mesh):
   
    r,phi = cart2pol(curve[:,1]-c_mesh,curve[:,0]-c_mesh)
    
    s_phi = phi; s_phi[1:] = phi[1:] + 0.0001
    sign_change=np.where((s_phi[1:]*s_phi[:-1])<0)[0]   
    
    phi1=phi[:sign_change[0]+1]+360
    rho1=r[:sign_change[0]+1] 
       
    phi1=np.flip(phi1,axis=0)
    rho1=np.flip(rho1,axis=0)
    
    
    
    phi2=phi[sign_change[0]+1:] 
    rho2=r[sign_change[0]+1:]
    
    phi2=np.flip(phi2,axis=0)
    rho2=np.flip(rho2,axis=0)

    nnphi = np.append(phi2,phi1)
    nnr = np.append(rho2,rho1)
    
    interp_phi = np.array(range(0,360,6))
    
    
    f=interp1d(nnphi,nnr,bounds_error=False,fill_value=tuple([nnr[0],nnr[-1]])) 
    interp_rho=f(interp_phi)
    
    x,y=pol2cart(interp_rho,interp_phi)
    curve=np.array([y+c_mesh,x+c_mesh]).T
    return curve
 