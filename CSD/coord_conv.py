# -*- coding: utf-8 -*-

import numpy as np

def cart2pol(x,y):
    rho=np.sqrt(x**2+y**2)
    phi=np.arctan2(y,x)
    phi=phi*(180/np.pi)
    return(rho,phi)

def pol2cart(rho, phi):
    phi=phi*(np.pi/180)
    x=rho*np.cos(phi)
    y=rho*np.sin(phi)
    return(x,y)