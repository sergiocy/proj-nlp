# -*- coding: utf-8 -*-


from numpy import (array, dot, arccos, clip) 
from numpy.linalg import norm 

    
def compute_similarity_cosin(v1, v2):
    u = array(v1) 
    v = array(v2) 
    d = dot(u, v)
    c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle 
    angle = arccos(clip(c, -1, 1)) # if you really want the angle 

    return d, c, angle   