#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:27:09 2024

@author: francesco
"""

import numpy as np

def D2rho(D0, target_spin, spin_cutoff):
    '''
    calculate D from rho at Bn (Larsen et al. 2011, eq. (20))
    Takes as input rho as 1/MeV, and gives output D as eV
    target_spin is the spin of the target nucleus
    spin_cutoff is self-explanatory - calculate with robin
    '''
    factor = 2*spin_cutoff**2/((target_spin + 1)*np.exp(-(target_spin + 1)**2/(2*spin_cutoff**2)) + target_spin*np.exp(-target_spin**2/(2*spin_cutoff**2)))
    rho = factor/(D0*1e-6)
    return rho

def drho(target_spin, sig, dsig, D0, dD0, rho = None):
    '''
    Calculate the uncertainty in rho, given the input parameters. sig and dsig
    are the spin cutoff parameter and its uncertainty, respectively. Code taken
    from D2rho in the oslo_method_software package.
    '''
    
    alpha = 2*sig**2
    dalpha = 4*sig*dsig
    if target_spin == 0:
        y1a = (target_spin+1.0)*np.exp(-(target_spin+1.0)**2/alpha)
        y1b = (target_spin+1.)**2*y1a
        z1  = y1b
        z2  = y1a
    else:
        y1a = target_spin*np.exp(-target_spin**2/alpha)
        y2a = (target_spin+1.)*np.exp(-(target_spin+1.)**2/alpha)
        y1b = target_spin**2*y1a
        y2b = (target_spin+1.)**2*y2a
        z1  = y1b+y2b
        z2  = y1a+y2a
    u1 = dD0/D0
    u2 = dalpha/alpha
    u3 = 1-z1/(alpha*z2)
    if rho == None:
        rho = D2rho(D0, target_spin, sig)
    return rho*np.sqrt(u1**2 + u2**2*u3**2)

def chisquared(theo,exp,err,DoF=1,method = 'linear',reduced=True):
    if (len(theo) == len(exp)) & (len(exp) == len(err)):
        chi2=0
        if method == 'linear':
            for i in range(len(theo)):
                chi2+=((theo[i] - exp[i])/err[i])**2
            if reduced:
                return chi2/(len(theo)-DoF)
            else:
                return chi2
        elif method == 'log':
            for i in range(len(theo)):
                expmax = exp[i] + err[i]/2
                expmin = exp[i] - err[i]/2
                chi2+=(np.log(theo[i]/exp[i])/np.log(expmax/expmin))**2
            if reduced:
                return chi2/(len(theo)-DoF)
            else:
                return chi2
        else:
            print('Method not known')
    else:
        print('Lengths of arrays not matching')
        
def import_Anorm_alpha(path):
    string = np.genfromtxt(path)
    Anorm, alpha = [x for x in string if np.isnan(x) == False]
    return Anorm, alpha

def import_Bnorm(path):
    arr = np.loadtxt(path, skiprows = 3)
    return arr.item()

def calc_errors_chis(lst, lab, graphic_function = 'find_chis', return_best_fit = False):
    xx = lst[4].x #all energy or temperature vectors are alike. Pick the 5th, but any would do
    val_matrix = np.zeros((xx.size,5))
    
    #find nld or gsf with least chi2-score
    lstargmin = np.nanargmin([el.chi2 for el in lst])
    lstmin = lst[lstargmin]
    
    graphic_function_s = graphic_function
    if graphic_function_s == 'find_chis':
        graphic_function = find_chis
    elif graphic_function_s == 'find_chis_interp':
        graphic_function = find_chis_interp
    
    for i, x in enumerate(xx):
        chis = []
        vals = []
        row = np.zeros(5)
        counterflag = 0
        
        #calc_errors_chis_core(y, isrho = False)
        for graph in lst:
            if not np.isnan(graph.y[i]):
                if x == xx[-1] and lab == 'nld':
                    chis.append(graph.rhoSn_chi2)
                else:
                    chis.append(graph.chi2)
                vals.append(graph.y[i])
                #it may be that all y[i] are nans. Do a check so that the code only saves data if there actually are values to analyze
                counterflag += 1
        if counterflag > 10:
            errmin, errmax = graphic_function(vals,chis)
            row[:] = [x, lstmin.y[i], errmin, errmax, lstmin.yerr[i]]
        else:
            row[:] = [x, np.nan, np.nan, np.nan, np.nan]
        val_matrix[i,:] = row[:]
    if return_best_fit:
        return val_matrix, lstmin
    else:
        return val_matrix
    
def find_chis_interp(vals, chis, iterations = 2):
    '''
    New, more precise algorithm than find_chis when this is not good enough. Potentially slower.
    concept: first, make an array of datapoints (e.g. "points") with vals as x and chis as y.
    Then sort these for increasing chi.
    '''
    points = np.c_[vals,chis]
    points = points[np.lexsort((points[:,1],points[:,0]))]
    chimin_index = np.argmin(points[:,1])
    
    vertices_less = points[0]
    vertices_less = np.expand_dims(vertices_less, axis=0)
    
    for i, point in enumerate(points):
        if point[1] < vertices_less[-1,1]:
            vertices_less = np.vstack((vertices_less, point))
        if i==chimin_index: #(just consider the points with vals less than chimin)
            break
    
    vertices_more = points[-1]
    vertices_more = np.expand_dims(vertices_more, axis=0)
    for i, point in enumerate(points[::-1]):
        if i==chimin_index: #(just consider the points with vals more than chimin)
            break
        if point[1] < vertices_more[-1,1]:
            vertices_more = np.vstack((vertices_more, point))

    def delete_points(vertices_input, invert):
        if invert:
            vertices = np.c_[vertices_input[:,0]*-1, vertices_input[:,1]]
        else:
            vertices = vertices_input
        delete_indexes = []
        x0 = vertices[0,0]
        y0 = vertices[0,1]
        for i, point in enumerate(vertices):
            if i == len(vertices)-2:
                break
            elif i == 0:
                pass
            else:
                x1 = vertices[i,0]
                y1 = vertices[i,1]
                x2 = vertices[i+1,0]
                y2 = vertices[i+1,1]
                
                if x2 == x1:
                    delete_indexes.append(i)
                elif x1 == x0:
                    pass
                else:
                    prev_steepness = (y1-y0)/(x1-x0)
                    next_steepness = (y2-y1)/(x2-x1)
                    if next_steepness < prev_steepness:
                        delete_indexes.append(i)
                    else:
                        x0 = vertices[i,0]
                        y0 = vertices[i,1]
        
        return np.delete(vertices_input, delete_indexes, 0)
        
        
    for i in range(iterations):
        vertices_less = delete_points(vertices_less, invert = False)  
        vertices_more = delete_points(vertices_more, invert = True)
    vertices = np.vstack((vertices_less, vertices_more[::-1]))
    chimin = np.min(vertices[:,1])
    for i, vertex in enumerate(vertices):
        if vertex[1] < (chimin+1):
            min1 = vertices[i-1,:]
            min2 = vertices[i,:]
            break
    for i, vertex in reversed(list(enumerate(vertices))):
        if vertex[1] < (chimin+1):
            max1 = vertices[i+1,:]
            max2 = vertices[i,:]
            break
    # y(x) = A + Bx
    Bmin = (min2[1]-min1[1])/(min2[0]-min1[0])
    Amin = min2[1]-Bmin*min2[0]
    Bmax = (max2[1]-max1[1])/(max2[0]-max1[0])
    Amax = max2[1]-Bmax*max2[0]
    # evaluate at Y = chimin + 1
    Y = chimin + 1
    Xmin = (Y-Amin)/Bmin
    Xmax = (Y-Amax)/Bmax
    return [Xmin,Xmax]

def find_chis(vals,chis):
    '''
    function taking as input all chi2-scores associated to the values for a single energy or temperature
    it finds where the function crosses the chi2+1 line
    '''
    whole_mat = np.c_[vals,chis]#np.vstack((vals,chis)).T
    chimin = np.min(chis)
    lower_mat = whole_mat[chis<=(chimin+1)]
    upper_mat = whole_mat[(chis>(chimin+1)) & (chis<(chimin+3))]
    
    min1 = upper_mat[upper_mat[:,0]==np.min(upper_mat[:,0])][0]
    min2 = lower_mat[lower_mat[:,0]==np.min(lower_mat[:,0])][0]
    max1 = lower_mat[lower_mat[:,0]==np.max(lower_mat[:,0])][0]
    max2 = upper_mat[upper_mat[:,0]==np.max(upper_mat[:,0])][0]
    
    # y(x) = A + Bx
    Bmin = (min2[1]-min1[1])/(min2[0]-min1[0])
    Amin = min2[1]-Bmin*min2[0]
    Bmax = (max2[1]-max1[1])/(max2[0]-max1[0])
    Amax = max2[1]-Bmax*max2[0]
    
    # evaluate at Y = chimin + 1
    Y = chimin + 1
    Xmin = (Y-Amin)/Bmin
    Xmax = (Y-Amax)/Bmax
    return [Xmin,Xmax]

def clean_valmatrix(val_matrix):
    return val_matrix[~np.isnan(val_matrix).any(axis=1)]