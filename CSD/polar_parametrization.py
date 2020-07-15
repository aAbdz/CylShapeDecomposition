# -*- coding: utf-8 -*-

import numpy as np
from coord_conv import cart2pol, pol2cart

def polar_parametrization(curve, c_mesh):

    r,phi = cart2pol(curve[:,1]-c_mesh,curve[:,0]-c_mesh)
    
    s=phi<0 
    s_inx=np.where(s)[0]
    s_inx=s_inx[np.argmin(abs(phi[s_inx]))]

    nphi=np.append(phi[s_inx:],phi[:s_inx])
    nr=np.append(r[s_inx:],r[:s_inx]) 
    
    for i in range(3):    
        d_ang=np.diff(nphi)
        d_ang=np.append(nphi[0],d_ang)
        cw_direction=np.sign(d_ang)>=0   
        if sum(cw_direction)>(len(cw_direction)/2):
            'error'
        
        else:
            cw_dir=np.where(cw_direction)[0]
            cw_dir=cw_dir[abs(d_ang[cw_direction])<350]           
            nr=np.delete(nr,cw_dir)
            nphi=np.delete(nphi,cw_dir)
    
    sign_change=np.where((nphi[1:]*nphi[:-1])<0)[0]   
    if len(sign_change)>1:  
        over_st_point=np.where(nphi<nphi[0])[0]
        over_st_point=over_st_point[over_st_point>sign_change[1]]
       
        nr=np.delete(nr,over_st_point)
        nphi=np.delete(nphi,over_st_point)

    x,y=pol2cart(nr,nphi)
    curve=np.array([y+c_mesh,x+c_mesh]).T
    return curve