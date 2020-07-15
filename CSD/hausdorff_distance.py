# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(curve1,curve2,n_sampling):
    
    s1=np.floor(np.linspace(0,len(curve1)-1,n_sampling)).astype(int)
    s2=np.floor(np.linspace(0,len(curve2)-1,n_sampling)).astype(int) 
    u=curve1[s1]
    v=curve2[s2]
    curve_dist=max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return curve_dist