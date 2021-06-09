# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 23:38:29 2021

@author: Admin
"""
import numpy as np
import random as random
import cv2

def draw_boxes(mask,objects,colors):
    color_mask = np.zeros((mask.shape[0],mask.shape[1],3))
    for j,(objectID, centroid) in enumerate(objects.items()):
        color_mask[mask==centroid[-1],:] = colors[objectID]
        
        text = "ID_{}".format(objectID)
        cv2.putText(color_mask, text, (centroid[0], centroid[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[objectID], 1)
        cv2.rectangle(color_mask, (centroid[0], centroid[1]), (centroid[0]+centroid[2], centroid[1]+centroid[3]), colors[objectID], 1)
        
    return color_mask

def generate_colors(N):
    colors = []
    image  = []
    for i in range(N):
        c = np.array([random.random(),random.random(),random.random()])
        colors.append(c)
        
        a = np.zeros((5,10,3))
        a[:,:,0] = c[0]
        a[:,:,1] = c[1]
        a[:,:,2] = c[2]
        image.append(a)
    image = np.concatenate(image,axis=0)
    return colors,image