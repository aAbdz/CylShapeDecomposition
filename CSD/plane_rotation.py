# -*- coding: utf-8 -*-

import numpy as np

def rotate_vector(vector, rot_mat):
    
    "rotating a vector by a rotation matrix"
    rotated_vec = np.dot(vector,rot_mat)
    return rotated_vec


def rotation_matrix_3D(vector, theta):
    
    """counterclockwise rotation about a unit vector by theta radians using
    Euler-Rodrigues formula: https://en.wikipedia.org/wiki/Euler-Rodrigues_formula"""

    a=np.cos(theta/2.0)
    b,c,d=-vector*np.sin(theta/2.0)
    aa,bb,cc,dd=a**2, b**2, c**2, d**2
    bc,ad,ac,ab,bd,cd=b*c, a*d, a*c, a*b, b*d, c*d
    
    rot_mat=np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rot_mat

def unit_normal_vector(vec1, vec2):
    
    n = np.cross(vec1, vec2)
    if np.array_equal(n, np.array([0, 0, 0])):
        n = vec1
        
    s = max(np.sqrt(np.dot(n,n)), 1e-5)
    n = n/s
    return n

def angle(vec1, vec2):
    
    theta=np.arccos(np.dot(vec1,vec2) / (np.sqrt(np.dot(vec1,vec1)) * np.sqrt(np.dot(vec2, vec2))))
    return theta
