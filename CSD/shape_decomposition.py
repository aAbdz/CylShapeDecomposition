# -*- coding: utf-8 -*-

import numpy as np
import plane_rotation as pr
from scipy.interpolate import RegularGridInterpolator as rgi
from unit_tangent_vector import unit_tangent_vector
from hausdorff_distance import hausdorff_distance
from skimage.measure import label, regionprops
from skeleton_decomposition import skeleton_main_branch
import pylab as plt
from polar_parametrization import polar_parametrization
from polar_interpolation import polar_interpolation
import skfmm
from scipy.interpolate import interp1d
from matplotlib import path
from coord_conv import cart2pol


def tangent_planes_to_zone_of_interest(cropAx, parametrized_skel, 
                                             s_inx, e_inx, g_radius, g_res, shift_impose, direction, H_th):
    
    p_inx, p_bound, p_shiftX, p_shiftY = s_inx, [], 0 , 0 
    sz = cropAx.shape
    x, y = np.mgrid[-g_radius:g_radius:g_res, -g_radius:g_radius:g_res]
    z = np.zeros_like(x)
    
    c_mesh = (2*g_radius)/(2*g_res)
    
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T
    
    tangent_vecs = unit_tangent_vector(parametrized_skel)   
    interpolating_func = rgi((range(sz[0]),range(sz[1]),range(sz[2])), 
                                cropAx,bounds_error=False,fill_value=0)
    
    cent_ball = (x**2+y**2)<g_res*1
    count = 1
    while s_inx != e_inx:
        
        #print s_inx
        
        point = parametrized_skel[s_inx]        
        utv = tangent_vecs[s_inx]
        
        if np.array_equal(utv, np.array([0, 0, 0])):
            s_inx = s_inx+direction
            continue

        rot_axis = pr.unit_normal_vector(utv, np.array([0,0,1]))
        theta = pr.angle(utv, np.array([0,0,1]))
        rot_mat = pr.rotation_matrix_3D(rot_axis, theta)
        rotated_plane = np.squeeze(pr.rotate_vector(xyz, rot_mat))        
        cross_section_plane = rotated_plane+point            
        cross_section = interpolating_func(cross_section_plane)
        bw_cross_section = cross_section>=0.5
        bw_cross_section = np.reshape(bw_cross_section, x.shape)
        label_cross_section, nn = label(bw_cross_section, neighbors=4, return_num=True)
        main_lbl = np.unique(label_cross_section[cent_ball])
        main_lbl = main_lbl[np.nonzero(main_lbl)]
    
        if len(main_lbl)!=1:
            s_inx = s_inx+direction
            continue
        
        bw_cross_section = label_cross_section==main_lbl
        
        nz_X = np.count_nonzero(np.sum(bw_cross_section, axis=0))
        nz_Y = np.count_nonzero(np.sum(bw_cross_section, axis=1))      
        if (nz_X<4) | (nz_Y<4):
            s_inx = s_inx+direction
            continue
        
        if shift_impose:
            
            props = regionprops(bw_cross_section.astype(np.int))            
            y0, x0 = props[0].centroid
    
            shiftX = np.round((x.shape[0]/2)-x0).astype(np.int)
            shiftY = np.round((x.shape[1]/2)-y0).astype(np.int)
            
            p = max(abs(shiftX), abs(shiftY))
            
            if p != 0:
                
                bw_cross_section = np.pad(bw_cross_section, p, mode='constant')
    
                bw_cross_section = np.roll(bw_cross_section,shiftY,axis=0)
                bw_cross_section = np.roll(bw_cross_section,shiftX,axis=1)
            
                bw_cross_section = bw_cross_section[p:-p, p:-p]
                
        
        label_cross_section, nn = label(bw_cross_section, neighbors=4, return_num=True)
        if nn != 1:
            main_lbl = np.unique(label_cross_section[cent_ball])
            main_lbl = main_lbl[np.nonzero(main_lbl)]
        
            if len(main_lbl)!=1:
                s_inx = s_inx+direction
                continue
            bw_cross_section = label_cross_section==main_lbl
        
        
        bound = boundary_parametrization(bw_cross_section)
        
        if test_boundary_parametrization(bound, c_mesh) == False:
            s_inx = s_inx+direction
            continue
            
        
        #fig, ax=plt.subplots() 
        #ax.plot(bound[:,1], bound[:,0], '-', linewidth=2, color='black')
        if count==1:
            m_curve = mean_curve(bound, bound, 2, c_mesh, 0)            
            max_radius = np.max(np.sum((m_curve-np.array(x.shape)/2)**2, axis=1)**0.5)
            
            p_inx = s_inx
            p_bound =  bound
            p_shiftX = shiftX
            p_shiftY = shiftY
            
            count = count+1
            s_inx = s_inx+direction          
        else:
            H_dist = hausdorff_distance(bound, m_curve, len(m_curve))
            d_ratio = np.true_divide(H_dist, (H_dist+max_radius))                
                
            if d_ratio<H_th:
                m_curve = mean_curve(bound, m_curve, count, c_mesh, 0)
                max_radius = g_res*np.max(np.sum((m_curve-np.array(x.shape)/2)**2, axis=1)**0.5)
                
                p_inx = s_inx
                p_bound =  bound
                p_shiftX = shiftX
                p_shiftY = shiftY
                    
                count = count+1
                s_inx = s_inx+direction

            else:
                break
    return p_inx, p_bound, p_shiftX, p_shiftY
    
    
def test_boundary_parametrization(bound, c_mesh):
    
    flag = True
    p_bound = polar_parametrization(bound, c_mesh)
    r,phi = cart2pol(p_bound[:,1]-c_mesh, p_bound[:,0]-c_mesh)
    s_phi = phi; s_phi[1:] = phi[1:] + 0.0001
    sign_change = np.where((s_phi[1:]*s_phi[:-1])<0)[0]   
    if len(sign_change) == 0:
        flag = False
    return flag   
   

def find_junction_in_skeleton(parametrized_skel, junction_coordinate):
    
    flag = False
    main_junction_coordinates = []
    for jc in junction_coordinate:
        if jc in parametrized_skel:
            flag = True
            main_junction_coordinates.append(jc)
    return flag, main_junction_coordinates


def crop_image(bw, point, rect):
    
    sR = max(point[0]-rect[0], 0)
    eR = min(point[0]+rect[0], bw.shape[0])
    
    sC = max(point[1]-rect[1], 0)
    eC = min(point[1]+rect[1], bw.shape[1])
    
    sH = max(point[2]-rect[2], 0)
    eH = min(point[2]+rect[2], bw.shape[2])
       
    bbw = bw[sR:eR,sC:eC,sH:eH]
    new_point = point - np.array([sR,sC,sH])
    return bbw, new_point
    
def zone_of_interest(cropAx, parametrized_skel, junction_coordinate):
    
    mean_junction_coordinate = np.mean(junction_coordinate, axis=0)
    dist2junc = np.sqrt(np.sum((parametrized_skel-mean_junction_coordinate)**2, axis=1))
    min_dist_inx = np.argmin(dist2junc)
    min_dist = np.min(dist2junc)
    l_skel = len(parametrized_skel)-1
            
    try:
        sd = max(50, min_dist); rect = np.array([sd,sd,sd])
        
        while True:
            maximal_ball_radius_lb = maximal_inner_sphere(cropAx, parametrized_skel, junction_coordinate, rect)
            if maximal_ball_radius_lb < sd:
                break
            else:
                rect = rect+10
                
    except:
        maximal_ball_radius_lb = 2
    
    maximal_ball_radius_lb = max(maximal_ball_radius_lb, min_dist, 4) 
            
    
    ss_inx = np.array([0, l_skel])
    ee_inx = ss_inx 
    ub = 20; lb = 2
    coeff_ub = np.linspace(ub, lb, 40)
    
    
    maximal_ball_radius_ub = lb*maximal_ball_radius_lb
    e_inx = dist2junc > maximal_ball_radius_ub
    e_inx = np.where(np.logical_xor(e_inx[1:], e_inx[:-1]))[0]
    
    if len(e_inx) == 2:
        if e_inx[0] <= min_dist_inx  <= e_inx[1]:
            ee_inx = e_inx
    elif len(e_inx) == 1:
        if e_inx > min_dist_inx:
            ee_inx[1] = e_inx
        else:
            ee_inx[0] = e_inx
  
  
    for coeff in coeff_ub:
        maximal_ball_radius_ub = coeff*maximal_ball_radius_lb
        s_inx = dist2junc > maximal_ball_radius_ub
        s_inx = np.where(np.logical_xor(s_inx[1:], s_inx[:-1]))[0]
    
        if len(s_inx) == 2:
            if s_inx[0] <= min_dist_inx  <= s_inx[1]:
                ss_inx = s_inx
                break
    
    if len(s_inx) == 1:
        if s_inx > min_dist_inx:
            ss_inx[1] = s_inx
        else:
            ss_inx[0] = s_inx
            
            
    if ~(ss_inx[0] <= ee_inx[0] <= min_dist_inx):
        ss_inx[0] = 0
        ee_inx[0] = 0
        
    if ~(min_dist_inx <= ee_inx[1] <= ss_inx[1]):
        ss_inx[1] = l_skel
        ee_inx[1] = l_skel
            
    if ~(ss_inx[0] <= ee_inx[0] <= min_dist_inx <= ee_inx[1] <= ss_inx[1]):
        ss_inx = [0, l_skel] 
        ee_inx = ss_inx 
        
    return ss_inx, ee_inx


def obj_ends_conditions(dist2junc):
    
    if dist2junc[0] < dist2junc[-1]:
        s_dist = min(max(dist2junc), 20)
        s_inx = [0]
        s_inx.append(np.argmin(np.abs(dist2junc - s_dist)))        
        s_dist = min(max(dist2junc), 5)
        e_inx = [0]
        e_inx.append(np.argmin(np.abs(dist2junc - s_dist)))  
        
    else:
        s_dist = min(max(dist2junc), 20)
        s_inx = [np.argmin(np.abs(dist2junc - s_dist))]
        s_inx.append(len(dist2junc)-1)
        s_dist = min(max(dist2junc), 5)
        e_inx = [np.argmin(np.abs(dist2junc - s_dist))]
        e_inx.append(len(dist2junc)-1)
    
    return s_inx, e_inx


def maximal_inner_sphere(cropAx, parametrized_skel, junction_coordinate, rect):
    
    mean_junction_coordinate = np.mean(junction_coordinate, axis=0)
    f_jc = tuple(np.floor(mean_junction_coordinate).astype(np.int))

    if cropAx[f_jc] != 1:
        
        dist2junc = np.sqrt(np.sum((parametrized_skel-mean_junction_coordinate)**2, axis=1))
        min_dist_inx = np.argmin(dist2junc)
                
        l = min(min_dist_inx, len(dist2junc)-min_dist_inx)

        for i in range(l):
            f_jc = parametrized_skel[min_dist_inx+i]            
            f_jc = tuple(np.floor(f_jc).astype(np.int))                     
            if cropAx[f_jc] == 1:
                break
            else:
                f_jc = parametrized_skel[min_dist_inx-i] 
                f_jc = tuple(np.floor(f_jc).astype(np.int))
                if cropAx[f_jc] == 1:
                    break

    crop_obj, njc = crop_image(cropAx, f_jc, rect)
    
    D = skfmm.distance(crop_obj)   
    boundary = ((D!=0)==(D<=1))
    
    im_one = np.ones_like(crop_obj)   
    im_one[tuple(njc)] = 0
    D = skfmm.travel_time(im_one, crop_obj)
    
    dist_on_boundary = D[boundary]
    maximal_ball_radius_lb = np.min(dist_on_boundary)
    
    return maximal_ball_radius_lb


def corresponding_skel(im,final_skeleton,main_branches):
    
    c_skel=[]   
    for i in main_branches:
        main_branch=np.floor(final_skeleton[i]).astype(np.int)
        count=0
        for coord in main_branch:
            if im[tuple(coord)]==1:
                count += 1
        if np.true_divide(count,len(main_branch))>0.8:
            c_skel.append(final_skeleton[i])
    return c_skel


def detect_main_obj(obj, corrected_skeleton):
    
    count = 0
    flag = False
    f_skel = np.floor(corrected_skeleton).astype(np.int)
    f_skel = np.unique(f_skel, axis=0)
    for point in f_skel:
        if obj[tuple(point)]:
            count = count+1           
    if np.true_divide(count, len(f_skel)) > 0.6:
        flag = True
    return flag


def boundary_parametrization(bw):
    
    sz=np.array(bw.shape)+2
    p_bw=np.zeros(sz)
    p_bw[1:-1,1:-1]=bw
    
    f_0=np.array(np.unravel_index(np.argmax(p_bw),sz))
    Cor=f_0
    nCor=f_0+1
    bound=[f_0]
    x=[0,-1, 0,1]
    y=[1, 0,-1,0]
    move=np.array((x,y)).T
    direc=2
    while np.any(nCor!=f_0):
        Temp_dir=np.mod(direc+3,4)
        for i in range(4):
            nCor=Cor+move[Temp_dir]      
            if p_bw[tuple(nCor)]:
                direc=Temp_dir
                Cor=nCor
                bound.append(nCor)  
                break
            Temp_dir=Temp_dir+1
            if Temp_dir==4:
                Temp_dir=0
    bound=np.array(bound)-1
    return bound

    
def mean_curve(curve1, curve2, num_samp, c_mesh, vis):

    curve1 = polar_parametrization(curve1, c_mesh)
    curve1 = polar_interpolation(curve1, c_mesh)
    
    if num_samp==2:
        curve2=polar_parametrization(curve2, c_mesh)  
        curve2=polar_interpolation(curve2, c_mesh)
    
    m_curve=np.true_divide(np.sum((curve1,(num_samp-1)*curve2),axis=0),num_samp)
    
    if vis:
        fig, ax=plt.subplots() 
        ax.plot(curve1[:,1], curve1[:,0], '-', linewidth=2, color='black')
        ax.plot(curve2[:,1], curve2[:,0], '-', linewidth=2, color='red')
        ax.plot(m_curve[:,1], m_curve[:,0], '-', linewidth=2, color='blue')
        ax.set_xlim([0,120])
        ax.set_ylim([120,0])
    return m_curve

def interpolated_super_tube(curve1, curve2, num_steps):
    
    polygons = []
    curve1_coeff = np.linspace(1,0,num_steps)
    curve2_coeff = 1 - curve1_coeff    
    for i in range(len(curve1_coeff)):
        polygon = np.sum((curve1_coeff[i]*curve1, curve2_coeff[i]*curve2), axis=0)
        
        #fig, ax=plt.subplots() 
        #ax.plot(polygon[:,1], polygon[:,0], '-', linewidth=2, color='black')
        #ax.set_xlim([0,120])
        #ax.set_ylim([120,0])
        
        polygons.append(polygon)
    return polygons


def curve_interp(curve,c_sampling):

    sz=curve.shape
    interpolated_curve=np.empty((c_sampling,0))
    x=range(sz[0])
    xnew=np.linspace(0,len(curve)-1,c_sampling)
    for i in range(sz[1]): 
        y=curve[:,i]
        f=interp1d(x,y) 
        interp_f=np.expand_dims(f(xnew),axis=1)
        interpolated_curve=np.append(interpolated_curve,interp_f,axis=1)
    return interpolated_curve


def object_decomposition(obj, interpolated_skel, filled_cs, g_radius=15, g_res=0.25):
    
    sz = obj.shape
    x, y = np.mgrid[-g_radius:g_radius:g_res, -g_radius:g_radius:g_res]
    z = np.zeros_like(x)
    xyz = np.array([np.ravel(x), np.ravel(y), np.ravel(z)]).T
    tangent_vecs = unit_tangent_vector(interpolated_skel)   
    
    for i in range(len(interpolated_skel)):
        point = interpolated_skel[i]
        utv = tangent_vecs[i]
        
        if np.array_equal(utv, [0, 0, 0]):
            continue
        
        rot_axis = pr.unit_normal_vector(utv, np.array([0,0,1]))
        theta = pr.angle(utv, np.array([0,0,1]))
        rot_mat = pr.rotation_matrix_3D(rot_axis, theta)
        rotated_plane = pr.rotate_vector(xyz, rot_mat)
        cross_section_plane = rotated_plane+point
        
        cs = np.ravel(filled_cs[i])
        
        discrete_coordinates = np.round(cross_section_plane).astype(np.int)
        for ii in range(len(discrete_coordinates)):
            inx = discrete_coordinates[ii]
            if np.all(inx>=0) and np.all(inx<=np.array(sz)-1):
                if cs[ii]==0:
                    obj[tuple(inx)] = 0
                else:
                    obj[tuple(inx)] = 1
    return obj


def filling_cross_sections(st_cross_sections, g_radius, g_res):
    
    ep = np.array(2*g_radius/g_res, dtype=np.int)
    x, y = np.mgrid[0:ep, 0:ep]    
    filled_cs = []
    for cs in st_cross_sections:
        p = path.Path(cs)
        f_cs = p.contains_points(np.hstack((np.ravel(x)[:,np.newaxis], np.ravel(y)[:,np.newaxis])))
        f_cs = np.reshape(f_cs, (ep,ep))
        filled_cs.append(f_cs)
    return filled_cs


def junction_correction(cropAx, parametrized_skel, main_junction_coordinates, 
               g_radius, g_res, H_th, shift_impose, Euler_step_size):
            
    s_inxs, e_inxs = zone_of_interest(cropAx, parametrized_skel, main_junction_coordinates)
    
    s_inx = s_inxs[0]
    e_inx = e_inxs[0]    
    p_inx1, bound1, shiftX1, shiftY1 = tangent_planes_to_zone_of_interest(cropAx, parametrized_skel,
                                                                                s_inx, e_inx, g_radius, g_res, shift_impose, +1, H_th)
    point1 = parametrized_skel[p_inx1]
    
    s_inx = s_inxs[1]
    e_inx = e_inxs[1]    
    p_inx2, bound2, shiftX2, shiftY2 = tangent_planes_to_zone_of_interest(cropAx, parametrized_skel, 
                                                                                s_inx, e_inx, g_radius, g_res, shift_impose, -1, H_th)
    point2 = parametrized_skel[p_inx2]
    
    c_mesh = (2*g_radius)/(2*g_res)
    
    if len(bound1) == 0:
        curve1 = c_mesh * np.ones((c_mesh,2), dtype=np.int)
    else:
        curve1 = polar_parametrization(bound1, c_mesh)
        curve1 = polar_interpolation(curve1, c_mesh)
    
    if len(bound2) == 0:
        curve2 = c_mesh * np.ones((c_mesh,2), dtype=np.int)
    else:    
        curve2 = polar_parametrization(bound2, c_mesh)  
        curve2 = polar_interpolation(curve2, c_mesh)
    
    if shift_impose:
        curve1 = curve1 - np.array([shiftY1, shiftX1])
        curve2 = curve2 - np.array([shiftY2, shiftX2])
        
    num_steps = np.floor(np.sqrt(np.sum((point2-point1)**2)) / Euler_step_size).astype(np.int)
    st_cross_sections = interpolated_super_tube(curve1, curve2, num_steps)
    interpolated_skel = interpolated_super_tube(point1, point2, num_steps)
        
    corrected_skeleton = parametrized_skel[:p_inx1]
    corrected_skeleton = np.append(corrected_skeleton, interpolated_skel, axis=0)
    corrected_skeleton = np.append(corrected_skeleton, parametrized_skel[p_inx2+1:], axis=0)
        
    return st_cross_sections, interpolated_skel, corrected_skeleton


def object_analysis(obj, skel):
    
    decomposed_objs = []
    decomposed_skeletons = []
        
    sub_skeletons = skeleton_main_branch(skel)
        
    for s in sub_skeletons:
        junction_coordinates = s['dec_nodes']
        sub_skeleton = s['skeleton']
        
        rec_obj = obj.copy()
        for junction_coordinate in junction_coordinates:
            
            st_cross_sections, interpolated_skel, sub_skeleton = junction_correction(rec_obj, sub_skeleton, junction_coordinate, 
                                                                   g_radius=15, g_res=0.25, H_th=0.7, shift_impose=1, Euler_step_size=0.5)
            
            
            interpolated_skel = np.array(interpolated_skel)               
            filled_cs = filling_cross_sections(st_cross_sections, g_radius=15, g_res=0.25)                
            rec_obj = object_decomposition(rec_obj, interpolated_skel, filled_cs, g_radius=15, g_res=0.25)
        
            labeled_obj = label(rec_obj, neighbors=4)
            for region in regionprops(labeled_obj):
                if region.area>=len(sub_skeleton):                    
                    dec_obj = np.zeros(labeled_obj.shape, dtype=np.bool)
                    for coordinates in region.coords:
                        dec_obj[coordinates[0], coordinates[1], coordinates[2]] = True
                    
                    if detect_main_obj(dec_obj, sub_skeleton):  
                        decomposed_objs.append(dec_obj)
                        decomposed_skeletons.append(sub_skeleton)
                                                
                        break
                    
                    
    return decomposed_objs, decomposed_skeletons
    
    












