# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:05:10 2022

@author: Pablo Garcia
"""

import numpy as np
import copy
#%%

def cube(entry,
         labels,
         limits,
         contrast,
         isowhere = ["front", "front", "front"],
         levels=100,
         display=True,
         ax = None,
         frame = True
         ):
    """
    entry : entry object.
    labels : (list) whether to have entry.angles ("angles"), entry.scans ("scans") 
            or entry.ebins ("ebins") for x, y and z axes in that order i.e.
            [xlabel, ylabel, zlabel].
    limits : (list) coordinate limits for x, y and z axes in that order i.e.
            [xlims, ylims, zlims].
    contrast : (list) factors between 0 and 1 multiplying the maximum value of 
            the data set for vmin and vmax argument, in that order i.e.
            [vmin factor, vmax factor]
    isowhere : (list, optional) whether to show the surfaces at the front ("front"), 
            back ("back") or on both ("both") sides of the cube. The default is 
            ["front", "front", "front"].
    levels : (int, optional) number of levels used in contourf.The default is 100.
    display : (True/False , optional) if True, displays the cube, if False the
            relevant data to produce the plots elsewhere is returned. Default is
            True.
    ax : (variable name, optional) axes to use for plots if display == True.
            Default is None.
    frame : (True/False, optional) if True, a frame is shown at the cube edges.
        Defualt is True.

    """
    
    # Dictionary for isowhere, gives indices
    dictionary = {"front":0, "back":1}
    
    # vmin, vmax
    emin = contrast[0]
    emax = contrast[1]
    
    # getting indices of a,s ane e in list [x, y,z]
    labels = np.asarray(labels)
    a = np.where(labels=="angles")[0][0]
    s = np.where(labels=="scans")[0][0]
    e = np.where(labels=="ebins")[0][0]
    
    # Initial whole axes data
    arrays = []
    arrays.insert(a, entry.angles)
    arrays.insert(s, entry.scans)
    arrays.insert(e, entry.ebins)
    
    # Indices corresponding to values closest to given limits
    Limits = []
    for i in range(3):
        idxlow = (np.abs(arrays[i]-limits[i][0])).argmin()
        idxhigh = (np.abs(arrays[i]-limits[i][1])).argmin()
        Limits.append([idxlow,idxhigh])

    # Final reduced array
    arrays = []
    arrays.insert(a, entry.angles[Limits[a][0]:Limits[a][1]+1])
    arrays.insert(s, entry.scans[Limits[s][0]:Limits[s][1]+1])
    arrays.insert(e, entry.ebins[Limits[e][0]:Limits[e][1]+1])
    
    # Reduced data set
    d = entry.data[Limits[s][0]:Limits[s][1]+1, 
                   Limits[a][0]:Limits[a][1]+1, 
                   Limits[e][0]:Limits[e][1]+1]
    
    # Indices corresponding to values closest to given limits in new arrays
    Limits = []
    for i in range(3):
        idxlow = (np.abs(arrays[i]-limits[i][0])).argmin()
        idxhigh = (np.abs(arrays[i]-limits[i][1])).argmin()
        Limits.append([idxlow,idxhigh])
        
    # Reduced meshgrid
    x, y, z = np.meshgrid(arrays[0], arrays[1], arrays[2])
    
    # Limits indices sorted as [front index, back index]
    sorted_limits = copy.deepcopy(Limits)
    sorted_limits[0].sort(reverse = True)
    sorted_limits[1].sort()
    sorted_limits[2].sort(reverse = True)
    
    # Iso surface values
    isoaval = sorted_limits[a]
    isosval = sorted_limits[s]
    isoeval = sorted_limits[e]

    isoxval = sorted_limits[0]
    isoyval = sorted_limits[1]
    isozval = sorted_limits[2]
    
    isoXYZ = [[], [], []]
    isoL = [[], [], []]
    isodir = [[], [], []]
    isooffset = [[], [], []]
    
    # Data surfaces order [x, y, z]
    isod = []  
    isod.insert(a, [d[:,isoaval[0],:],
                        d[:,isoaval[1],:]])
    isod.insert(s, [d[isosval[0],:,:],
                        d[isosval[1],:,:]])
    isod.insert(e, [d[:,:,isoeval[0]],
                        d[:,:,isoeval[1]]])
    
    # isox required coordinates (not data values)
    isoxY = y[:,0,:]
    isoxZ = z[:,0,:]
    
    # isoy required coordinates (not data values)
    isoyX = x[0,:,:]
    isoyZ = z[0,:,:]
    
    # isoz required coordinates (not data values)
    isozX = x[:,:,0]
    isozY = y[:,:,0]
    
    for i in range(2): # index 0 gives front view, 1 gives back view
        
        isoxX = isod[0][i]
        isoyY = isod[1][i]
        isozZ = isod[2][i]
        
        # Handles the different combinations of labels parameter by transposing
        if isoxX.shape != isoxY.shape or isoxX.shape != isoxZ.shape:
            isoxX = isoxX.T
        if isoyY.shape != isoyX.shape or isoyY.shape != isoyZ.shape:
            isoyY = isoyY.T
        if isozZ.shape != isozX.shape or isozZ.shape != isozX.shape:
            isozZ = isozZ.T
        
        # X, Y and Z for isox, isoy, isoz
        isoXYZ[0].extend([[isoxX, isoxY, isoxZ]])
        isoXYZ[1].extend([[isoyX, isoyY, isoyZ]])
        isoXYZ[2].extend([[isozX, isozY, isozZ]])
        
        # levels
        isoxL = np.linspace(np.min(isoxX),np.max(isoxX),levels)
        isoyL = np.linspace(np.min(isoyY),np.max(isoyY),levels)
        isozL = np.linspace(np.min(isozZ),np.max(isozZ),levels)
        isoL[0].extend([isoxL])
        isoL[1].extend([isoyL])
        isoL[2].extend([isozL])  
        
        # zdir
        isodir[0].extend(["x"])
        isodir[1].extend(["y"])
        isodir[2].extend(["z"]) 
        
        # offset
        isooffset[0].extend([arrays[0][isoxval[i]]])
        isooffset[1].extend([arrays[1][isoyval[i]]])
        isooffset[2].extend([arrays[2][isozval[i]]])
  
    # Displaying/ returning
    iso_return = []
    for i in range(3): # isox, isoy or isoz
        for j in range(2): # front view or back
            if isowhere[i] == "front" or isowhere[i] == "back":
                j = dictionary[isowhere[i]] # overwrites j value using dictonary
                                            # else, if both, for loop will access
                                            # front and back view.
            X = isoXYZ[i][j][0]
            Y = isoXYZ[i][j][1]
            Z = isoXYZ[i][j][2]
            levels = isoL[i][j]
            zdir = isodir[i][j]
            offset = isooffset[i][j]
            iso_return.extend([[[X,Y,Z], levels, zdir, offset, [x,y,z]]]) # relevant data
                                                                 # for each iso
            if display == True:
                ax.contourf(X = X, Y = Y, Z = Z, 
                            levels = levels, zdir = zdir, offset = offset,
                            vmin=emin*abs(d).max(), vmax=emax*abs(d).max(),
                            zorder = 0)
                ax.set(xlim = limits[0], 
                        ylim = limits[1], 
                        zlim = limits[2])
                
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
                ax.set_zlabel(labels[2])
                
                if frame == True: 
                    if isowhere[2] == "front" or isowhere[i] == "both":
                        color = "k"
                        lw = 0.5
                        alpha = 0.3
                        zorder = 1e4
                        ax.plot(x[0,:,0], 
                                y[isoyval[j],:,0], 
                                z[0,:,0]*0+np.max(limits[2]), 
                                color = color, alpha = alpha, lw = lw, zorder = zorder)
                        ax.plot(x[:,isoxval[j],0], 
                                y[:,0,0], 
                                z[:,0,0]*0+np.max(limits[2]), 
                                color = color, alpha = alpha, lw = lw, zorder = zorder)
                        ax.plot(x[0,0,:]*0+np.max(limits[0]), 
                                y[0,0,:]*0+np.min(limits[1]), 
                                z[0,0,:], 
                                color = color, alpha = alpha, lw = lw, zorder = zorder)
                        
                    if isowhere[2] == "back" or isowhere[i] == "both":
                        color = "k"
                        lw = 0.5
                        alpha = 0.3
                        zorder = 1e4
                        ax.plot(x[0,:,0], 
                                y[isoyval[j],:,0], 
                                z[0,:,0]*0+np.min(limits[2]), 
                                color = color, alpha = alpha, lw = lw, zorder = zorder)
                        ax.plot(x[:,isoxval[j],0], 
                                y[:,0,0], 
                                z[:,0,0]*0+np.min(limits[2]), 
                                color = color, alpha = alpha, lw = lw, zorder = zorder)
                        ax.plot(x[0,0,:]*0+np.min(limits[0]), 
                                y[0,0,:]*0+np.max(limits[1]), 
                                z[0,0,:], 
                                color = color, alpha = alpha, lw = lw, zorder = zorder)
                
            if isowhere[i] == "front" or isowhere[i] == "back":
                break # so that we don't do front and back view twice
            
    if display == False:
        return iso_return[0], iso_return[1], iso_return[2]
