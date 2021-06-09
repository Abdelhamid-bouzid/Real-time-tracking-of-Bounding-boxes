# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 23:41:40 2021

@author: Admin
"""
from utils import *
from CentroidTracker import CentroidTracker

ct           = CentroidTracker()
colors,image = generate_colors(50)
while True:
    # mask for creating a color mask (usally has the same shape as img)
    # boxes from predictor 
    objects = ct.update(boxes)
    draw_boxes(mask,objects,colors)