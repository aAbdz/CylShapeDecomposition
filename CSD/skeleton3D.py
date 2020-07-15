# -*- coding: utf-8 -*-

import numpy as np
import skfmm
import sys

def discrete_shortest_path(D,start_point):
       
    sz = D.shape   
    x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
    y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
    z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]
     
    path = [start_point]
    
    min_v = np.inf    
    while(min_v!=0):
              
        neighbor_inx = np.array((x,y,z)).T
        ngb = start_point + neighbor_inx

        valid_ngb_inx = np.where(np.all((np.all(ngb>=0,axis=1), np.all(ngb<sz,axis=1)), axis=0))
        ngb = ngb[valid_ngb_inx]
        
        ngb_value = [D[tuple(i)] for i in ngb]
        
        min_ind = np.argmin(ngb_value)
        min_v = ngb_value[min_ind]
    
        start_point = ngb[min_ind]
        path.append(start_point)
        
    path = np.array(path)
    return path





def pointmin(D):
    
    sz = D.shape
    max_D = np.max(D)
    Fx = np.zeros(sz)
    Fy = np.zeros(sz)
    Fz = np.zeros(sz)
    
    J = max_D * np.ones(np.array(sz)+2)
    J[1:-1,1:-1,1:-1] = D
    
    x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
    y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
    z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    #x = [1,-1, 0, 0, 0, 0]
    #y = [0, 0, 1,-1, 0, 0]
    #z = [0, 0, 0, 0, 1,-1]
    
    for i in range(26):       
        In = J[1+x[i]:1+sz[0]+x[i], 1+y[i]:1+sz[1]+y[i], 1+z[i]:1+sz[2]+z[i]]
        check = In<D
        D[check] = In[check]
        
        den = (x[i]**2 + y[i]**2 + z[i]**2)**0.5
        
        Fx[check]= x[i]/den
        Fy[check]= y[i]/den 
        Fz[check]= z[i]/den 
    return Fx, Fy, Fz






def euler_shortest_path(D,source_point,start_point,step_size):  
  
    Fx, Fy, Fz = pointmin(D)      
    Fx = -Fx
    Fy = -Fy
    Fz = -Fz
    
    itr = 0
    path = start_point
    while True:
        
        end_point = Euler_path(Fx,Fy,Fz,start_point,step_size)
        
        dist_endpoint_to_all = np.sum((source_point-end_point)**2,axis=1)**0.5
        distance_to_endpoint = min(dist_endpoint_to_all)
        
        if(itr>=10):
            Movement = np.sum((end_point-path[itr-10])**2)**0.5
        else:
            Movement = step_size+1
        
        if(np.all(end_point==0) or Movement<step_size):
            break
    
        itr = itr+1
        
        path = np.append(path,end_point,axis=0)
        
        if(distance_to_endpoint<10*step_size):
            source_inx = source_point[np.argmin(dist_endpoint_to_all)]
            path = np.append(path,np.array(source_inx,ndmin=2),axis=0)
            break
        
        start_point = end_point
    return path






def Euler_path(Fx,Fy,Fz,start_point,step_size):
    
    f_start_point = np.floor(start_point).astype(int)
    sz = Fx.shape
    
    x = [0, 0, 0, 0, 1, 1, 1, 1] 
    y = [0, 0, 1, 1, 0, 0, 1, 1]
    z = [0, 1, 0, 1, 0, 1, 0, 1]
    
    neighbor_inx = np.array((x,y,z)).T

    base = f_start_point + neighbor_inx
    base[base<0] = 0
    xbase=base[:,0]; xbase[xbase>=sz[0]]=sz[0]-1
    ybase=base[:,1]; ybase[ybase>=sz[1]]=sz[1]-1
    zbase=base[:,2]; zbase[zbase>=sz[2]]=sz[2]-1
    base=np.array((xbase,ybase,zbase)).T
    
    dist2f=np.squeeze(start_point-f_start_point)
    dist2c=1-dist2f

    perc = np.array((   dist2c[0]*dist2c[1]*dist2c[2],
                        dist2c[0]*dist2c[1]*dist2f[2],
                        dist2c[0]*dist2f[1]*dist2c[2],
                        dist2c[0]*dist2f[1]*dist2f[2],
                        dist2f[0]*dist2c[1]*dist2c[2],
                        dist2f[0]*dist2c[1]*dist2f[2],
                        dist2f[0]*dist2f[1]*dist2c[2],
                        dist2f[0]*dist2f[1]*dist2f[2]  ))
            

 
    gradient_valueX=[Fx[tuple(i)] for i in base]*perc
    gradient_valueY=[Fy[tuple(i)] for i in base]*perc 
    gradient_valueZ=[Fz[tuple(i)] for i in base]*perc 

    gradient_value=np.array((gradient_valueX,gradient_valueY,gradient_valueZ))

    sum_g=np.sum(gradient_value,axis=1)
    
    gradient=sum_g/((np.sum(sum_g**2)+0.000001)**0.5)

    end_point = start_point - step_size*gradient
    
    if (np.any(end_point<0) or end_point[0,0]>sz[0] or end_point[0,1]>sz[1] or end_point[0,2]>sz[2]):
        end_point=np.zeros((1,3))
    return end_point






def get_line_length(L):
    
    dist = np.sum(np.sum((L[1:] - L[:-1])**2,axis=1)**0.5)
    return dist






def organize_skeleton(skel_seg,length_th):
    
    final_skeleton = []
    
    n = len(skel_seg)
    end_points = np.zeros((n*2,3))
    
    l = 0
    for i in range(n):
        ss = skel_seg[i]
        l = max(l,len(ss))
        end_points[i*2] = ss[0]
        end_points[i*2+1] = ss[-1]

    connecting_distance = 2

    for i in range(n):

        ss = np.asarray(skel_seg[i])
        
        ex = np.reshape(end_points[:,0],(-1,1)); ex = np.repeat(ex,len(ss),axis=1)       
        sx = np.reshape(ss[:,0],(1,-1)); sx = np.repeat(sx,len(end_points),axis=0)
                   
        ey = np.reshape(end_points[:,1],(-1,1)); ey = np.repeat(ey,len(ss),axis=1)       
        sy = np.reshape(ss[:,1],(1,-1)); sy = np.repeat(sy,len(end_points),axis=0)

        ez = np.reshape(end_points[:,2],(-1,1)); ez = np.repeat(ez,len(ss),axis=1)
        sz = np.reshape(ss[:,2],(1,-1)); sz = np.repeat(sz,len(end_points),axis=0)        

        D = (ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2

        check = np.amin(D, axis=1) < connecting_distance
        check[i*2] = False
        check[i*2+1] = False
        
        cut_skel = [0,len(ss)]
        if(any(check)):
            for ii in range(len(check)):
                if(check[ii]):
                    line = D[ii]
                    min_ind = np.ma.argmin(line)
                    if((min_ind>2) and (min_ind<(len(line)-2))):
                        cut_skel.append(min_ind)
                        
        cut_skel = sorted(cut_skel)
        for j in range(len(cut_skel)-1):      
            skel_breaked_seg = ss[cut_skel[j]:cut_skel[j+1]]
            length_skel_seg = get_line_length(skel_breaked_seg)
            if(length_skel_seg >= length_th):
               final_skeleton.append(skel_breaked_seg)
               
    return final_skeleton





def skeleton(Ax):
    
    boundary_dist=skfmm.distance(Ax)
    
    source_point=np.unravel_index(np.argmax(boundary_dist), boundary_dist.shape)
    maxD=boundary_dist[source_point]
    
    speed_im=(boundary_dist/maxD)**1.5
    
    Ax=np.ones(Ax.shape)
    Ax[source_point]=0
    
    flag=True
    skeleton_segments=[]
    source_point = np.array(source_point,ndmin=2)
    while True:
        
        D=skfmm.travel_time(Ax,speed_im)
        end_point=np.unravel_index(np.ma.argmax(D), D.shape)
        max_dist=D[end_point]
        D=np.ma.MaskedArray.filled(D,max_dist)
        
        end_point = np.array(end_point,ndmin=2)
        shortest_line=euler_shortest_path(D,source_point,end_point,step_size=0.1)
        #shortest_line = discrete_shortest_path(D,end_point)
            
        line_length=get_line_length(shortest_line)
        print(line_length)

        if flag:
            length_threshold=min(40*maxD, 0.18*line_length)
            flag=False
        
        if(line_length<=length_threshold):
            break
        
        
        source_point=np.append(source_point,shortest_line,axis=0)
        
        skeleton_segments.append(shortest_line)
        
        shortest_line=np.floor(shortest_line).astype(int)
        
        for i in shortest_line:
            Ax[tuple(i)]=0
    
    if len(skeleton_segments)!=0:
        final_skeleton=organize_skeleton(skeleton_segments,length_threshold)
    else:
        final_skeleton=[]
    
    return final_skeleton

if __name__ == "__main__":    
    skeleton(sys.argv[1])











    
