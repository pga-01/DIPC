# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:14:11 2022

@author: Miguel Melchor
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from navarp.utils import navfile

#%%
def pad_with(vector, pad_width, iaxis, kwargs):  
     """
     Function in charge of doing the padding. Got it from numpy.pad() documentation.
     """
     padding_value = kwargs.get('padder', 0)  

def move(img,
    grid_dim,
    scans, 
    angles,
    overlaps=False,
    points=None,
    rec_crop=None,
    free_cut=None,
    reduce=False
):
    """
    img: (list) list of data entries.
    grid_dim: (list) dimension of working grid (angles x scans)
    reduce: (True/False) if True, reduces grid dimensions down to smallest possible size,
                         if False, grid dimensions are unchanged.
    overlaps: (list) which image lies on top of the other.
                     (0vs.1, 0vs.2, 1vs.2)
    rec_crop: (None or nested list/s): rectangular cropping of each image.
            if None no cropping is applied. 
            if nested list: first element: the image number which to crop
                    : second element: angles index, sets eveything to left to zero
                    : third element: angles index, sets everything to right to zero
                    : forth element: scans index: sets eveything below to zero
                    : fifth element: scans index: sets eveything above to zero
    free_cut: (None or nested list/s) cropping above/below or left/right of a 
            line between two specified points.
            if None no cropping is applied.
            if nested list: first element: the image number which to crop
                    : second element: (string) whether to crop above/below or left/right
                    : third element: angles index of first point
                    : forth element: scans index of first point
                    : fifth element: angles index of second point
                    : sixth element: scans index of second point                   
    scans: (list) amount of indices padded with zeros from the bottom of each image. 
                        At the same time the same amount of indices are removed from 
                        the top so scans allows movement of image along scans direction.
                        Order in list matches image number.
    angles: (list) amount of indices padded with zeros on left of each image. 
                        At the same time the same amount of indices are removed from 
                        the right so scans allows movement of image along angles direction.
                        Order in list matches image number.
    points: (None or nested list/s): matches two selected points in two images.
            if None no translation is applied.
            if nested list: Shape is:
                [[[img_num0, angle0, scan0], [img_num1, angle1, scan1]], [[etc.],[etc.]]]
                img_num0 will not move.
                img_num1 will move: (angle1, scan1) will match to (angle0, scan0)

    """
    scans = np.array(scans)
    angles = np.array(angles)
    
    # length of axes
    dim_scans = grid_dim[1]
    dim_angles = grid_dim[0]

    # matching points
    if points is not None:
        for p in points:
            if p is not None:

                fixed_angle = p[0][1]
                fixed_scan = p[0][2]

                moves_img = p[1][0]
                moves_angle = p[1][1]
                moves_scan = p[1][2]

                angles[moves_img]=fixed_angle - abs(moves_angle-angles[moves_img])
                scans[moves_img]=fixed_scan - abs(moves_scan-scans[moves_img])
    
    # reducing grid to min size. Recall rec_crop does not remove unwanted data,
    # but rather pads it so min size will include this padded data still.
    if reduce:
        scans_max = np.where(scans == max(scans))[0][0]
        angles_max = np.where(angles == max(angles))[0][0]
        dim_scans = scans[scans_max]+len(img[scans_max]) - min(scans)
        dim_angles = angles[angles_max] + len(img[angles_max][0]) - min(angles)
        
        scans = scans-min(scans)
        angles = angles-min(angles)
        
    # rectangular cropping
    if rec_crop is not None:
        for j in rec_crop:
            img[j[0]] = img[j[0]].copy()

            if j[1] is not None:
                img[j[0]][:,:j[1],:] = 0
            if j[2] is not None:
                img[j[0]][:,j[2]+1:,:] = 0
            if j[3] is not None:
                img[j[0]][:j[3],:,:] = 0
            if j[4] is not None:
                img[j[0]][j[4]+1:,:,:] = 0
                
    # other cropping methods
    if free_cut is not None:
        for j in free_cut:
            
            img[j[0]] = img[j[0]].copy()
            
            ai = min([j[2], j[4]])
            af = max(j[2], j[4])
            if ai == j[2]:
                si = j[3]
                sf = j[5]
            else:
                si = j[5]
                sf = j[3]
            
            if j[1]=="left" and ai==af: # vertical crop case
                if si != sf:
                    img[j[0]][si:sf+1, :ai, :] = 0
                else:
                    raise ValueError("scans_i and scans_f must be different but "
                                     "have value {}".format(j[3]))
                        
            elif j[1]=="right" and ai==af: #vertical crop case
                if si != sf:
                    img[j[0]][si:sf+1, ai+1:, :] = 0
                else:
                    raise ValueError("scans_i and scans_f must be different but "
                                     "have value {}".format(j[3]))    
                
            else: #slanted crop cases
                if ai == af:
                    raise ValueError("If crop is vertical choose left or right.")
                
                m = (sf-si)/(af-ai)
                c = si-m*ai 
            
                if abs(m) <= 1:
                                            
                    angles_arr = np.arange(ai, af)
                    scans_arr = m*angles_arr + np.zeros(len(angles_arr)) + c
                    scans_arr = np.rint(scans_arr).astype(int)  
                    
                elif abs(m) > 1:
                    
                    scans_arr = np.arange(si, sf+1)
                    angles_arr = (scans_arr - (np.zeros(len(scans_arr))+c))/m
                    angles_arr = np.rint(angles_arr).astype(int)
                    
                for i in range(len(angles_arr)):
                    
                    if j[1]=="left":
                        if si==sf:
                            raise ValueError("Crop is horizontal, set {} to " 
                                             "either above or below".format(j[1]))
                        else:
                            img[j[0]][scans_arr[i], :angles_arr[i], :] = 0
                    
                    if j[1]=="right":
                        if si==sf:
                            raise ValueError("Crop is horizontal, set {} to " 
                                             "either above or below".format(j[1]))
                        else:
                            img[j[0]][scans_arr[i], angles_arr[i]+1:, :] = 0
                        
                    if j[1]=="above":
                        # print(j[1])
                        if si==sf:
                            img[j[0]][si+1:, ai:af, :] = 0
                        elif ai==af:
                            raise ValueError("Crop is vertical, set {} to "
                                             "either left or right".format(j[1]))
                        else:
                            # print("scans_arr[i]", scans_arr[i])
                            # print("angles_arr[i]", angles_arr[i])
                            img[j[0]][scans_arr[i]+1:, angles_arr[i], :] = 0
                            
                    if j[1]=="below":
                        if si==sf:
                            img[j[0]][:j[3], ai:af, :] = 0
                        elif ai==af:
                            raise ValueError("Crop is vertical set {} to "
                                             "either left or right".format(j[1]))
                        else:
                            img[j[0]][:scans_arr[i], angles_arr[i], :] = 0
                                                                    
    # padding each image to fill grid_dim
    w = []
    for i in range(len(img)):
        # print("i", i)
        # print("(dim_scans-len(img[i])-scans[i])", (dim_scans-len(img[i])-scans[i]))
        # print(dim_scans, len(img[i]), scans[i]) 
        # print("(dim_angles-len(img[i][0])-angles[i])", (dim_angles-len(img[i][0])-angles[i]))
        # print(dim_angles, len(img[i][0]), angles[i])
        w.append(np.pad(img[i], [[scans[i], (dim_scans-len(img[i])-scans[i])],
                    [angles[i],(dim_angles-len(img[i][0])-angles[i])],
                    [0,0]], pad_with))

    Z = sum(w)

    # overlapping indices. Order in D is overlap between image 0&1,0&2,0&3... 1&2,1&3... 2&3.
    if overlaps:
        D = []
        for i in range(len(img)):
            j = 1
            while (i+j)<(len(img)): # ensures order is kept
                Z_pair = w[i]+w[i+j]
                D_pair = np.where((Z_pair!=w[i]) & (Z_pair!=w[i+j]))
                D.append(D_pair)
                j+=1
        
        # ensures overlapping indices correspond to the image number indicated in overlaps
        for i in range(len(overlaps)):
            ruler = overlaps[i] 
            Z[D[i]] = w[ruler][D[i]]

    else:
        pass
    
    return Z
