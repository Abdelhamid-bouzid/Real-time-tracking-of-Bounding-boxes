# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:02:57 2021

@author: Admin
"""
# import the necessary packages
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import config

class CentroidTracker():
    def __init__(self, maxDisappeared=1):
        
        self.nextObjectID   = 0
        self.objects        = OrderedDict()
        self.disappeared    = OrderedDict()
        self.maxDisappeared = maxDisappeared
        
    def register(self, centroid):
        # when registering an object we use the next available object ID to store the centroid
        self.objects[self.nextObjectID]     = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID                  += 1
        
        if len(self.objects) == 0:
            self.nextObjectID += 1
        else:
            objectIDs = set(self.objects.keys())
            dif       = set(range(0, max(objectIDs)+2)).difference(objectIDs)
            self.nextObjectID = next(iter(dif))
        
    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def update(self, rects):
        # check to see if the list of input bounding box rectangles is empty
        if rects.shape[0] == 0:
            
            self.nextObjectID   = 0
            self.objects        = OrderedDict()
            self.disappeared    = OrderedDict()
            return self.objects
        
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((rects.shape[0], 6), dtype="int")
        for i in range(rects.shape[0]):
            startX, startY, W, H, A, r = rects[i][0],rects[i][1],rects[i][2],rects[i][3],rects[i][4],rects[i][5]
            inputCentroids[i] = (startX, startY, W, H, A,r)
            
        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        
        # otherwise, are are currently tracking objects so we need to  try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs       = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            D      = dist.cdist(np.array(objectCentroids)[:,:4], inputCentroids[:,:4])
            D,unmatch_row,unmatch_col = self.mask_cost(objectCentroids,inputCentroids,D)
            
            rows, cols = self.linear_assignment(D)
            
            # in order to determine if we need to update, register, or deregister an object we need to keep track of which of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                
                if row in unmatch_row or col in unmatch_col:
                    continue
                
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row, set its new centroid, and reset the disappeared counter
                objectID                   = objectIDs[row]
                self.objects[objectID]     = inputCentroids[col]
                self.disappeared[objectID] = 0
                
                # indicate that we have examined each of the row and column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
                
            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            for col in unusedCols:
                self.register(inputCentroids[col])
            
            for row in unusedRows:
                objectID = objectIDs[row]
                self.deregister(objectID)
                
        # return the set of trackable objects
        return self.objects
    
    def linear_assignment(self,D):
        #rows    = D.min(axis=1).argsort()
        #cols    = D.argmin(axis=1)[rows]
        rows, cols = linear_sum_assignment(D)
        
        return rows, cols
    
    def mask_cost(self,objectCentroids,inputCentroids,D):
        
        ################################################### mask 1 ######################################################################
        a      = np.expand_dims(np.array(objectCentroids)[:,config.direction1],axis=1)
        b      = np.expand_dims(inputCentroids[:,config.direction1],axis=0)
        mask1  = np.abs(b-a)>20
        
        ################################################### mask 2 ######################################################################
        c      = np.expand_dims(np.array(objectCentroids)[:,config.direction2],axis=1)
        d      = np.expand_dims(inputCentroids[:,config.direction2],axis=0)
        s     = d-c
        mask2 = s>5
        
        mask = (mask1 | mask2)
        D[mask] = 300000
        
        unmatch_row = []
        for i in range(D.shape[0]):
            if np.all(D[i]==D[i][0]):
                unmatch_row.append(i)
                
        unmatch_col = []
        for i in range(D.shape[1]):
            if np.all(D[:,i]==D[:,i][0]):
                unmatch_col.append(i)
        return D,unmatch_row,unmatch_col