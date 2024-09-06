#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:51:12 2024

@author: francesco
"""

import numpy as np
import math
from dicts_and_consts import k_B

def import_ocl(path,a0,a1, no_errcol=False):
    '''
    import data generated with the oslo method software, and convert it to 
    something readable
    '''
    raw_matrix = np.loadtxt(path)
    
    if no_errcol:
        channels = int(raw_matrix.shape[0])
        polished_matrix = np.zeros((channels,2))
    else:
        channels = int(raw_matrix.shape[0]/2)
        polished_matrix = np.zeros((channels,3))
        
    limit = channels    
    for i, el in enumerate(raw_matrix):
        if i<limit:
            polished_matrix[i,0] = a0 + a1*i
            #if el == 0.0 or el == 'inf':
            if el == 'inf':
                polished_matrix[i,1] = np.nan
            else:
                polished_matrix[i,1] = el
        else:
            #if el == 0.0 or el == 'inf':
            if el == 'inf':
                polished_matrix[i-channels,2] = np.nan
            else:
                polished_matrix[i-channels,2] = el
    return polished_matrix

class gsf:
    '''
    Class for reading strength functions
    '''
    
    def __init__(self, path, a0, a1):
        self.path = path
        self.rawmat = import_ocl(path, a0, a1)
        self.x = self.rawmat[:,0]
        self.y = self.rawmat[:,1]
        self.yerr = self.rawmat[:,2]
            
    def clean_nans(self):
        clean_E = []
        clean_y = []
        clean_yerr = []
        for E, y, yerr in zip(self.x, self.y, self.yerr):
            if not math.isnan(y):
                clean_E.append(E)
                clean_y.append(y)
                clean_yerr.append(yerr)
        self.x = np.array(clean_E)
        self.y = np.array(clean_y)
        self.yerr = np.array(clean_yerr)
    
    def delete_point(self, position):
        self.clean_nans()
        self.y = np.delete(self.y, position)
        self.x = np.delete(self.x, position)
        self.yerr = np.delete(self.yerr, position)

class nld(gsf):
    '''
    Class for reading level densities
    '''
    def __init__(self, path, a0, a1):
        super().__init__(path, a0, a1)
        
    def add_rhoSn(self, Sn, rhoSn, drhoSn):
        self.x = np.append(self.x, Sn)
        self.y = np.append(self.y, rhoSn)
        self.yerr = np.append(self.yerr, drhoSn)
        
class ncrate():
    '''
    Small class for reading ncrates
    '''
    def __init__(self, path):
        self.rawmat = np.loadtxt(path)
        self.x = self.rawmat[:,0]
        self.y = self.rawmat[:,1]
        
class MACS():
    '''
    Small class for reading MACSs
    '''
    def __init__(self, path):
        self.rawmat = np.loadtxt(path)
        self.x = self.rawmat[:,0]*k_B
        self.y = self.rawmat[:,2]
        
        
class gsf_p:
    '''
    Class for reading strength functions
    '''
    
    def __init__(self, rawmat):
        self.rawmat = rawmat
        self.x = self.rawmat[:,0]
        self.y = self.rawmat[:,1]
        self.yerr = self.rawmat[:,2]
            
    def clean_nans(self):
        clean_E = []
        clean_y = []
        clean_yerr = []
        for E, y, yerr in zip(self.x, self.y, self.yerr):
            if not math.isnan(y):
                clean_E.append(E)
                clean_y.append(y)
                clean_yerr.append(yerr)
        self.x = np.array(clean_E)
        self.y = np.array(clean_y)
        self.yerr = np.array(clean_yerr)
    
    def delete_point(self, position):
        self.clean_nans()
        self.y = np.delete(self.y, position)
        self.x = np.delete(self.x, position)
        self.yerr = np.delete(self.yerr, position)

class nld_p(gsf_p):
    '''
    Class for reading level densities
    '''
    def __init__(self, rawmat):
        super().__init__(rawmat)
        
    def add_rhoSn(self, Sn, rhoSn, drhoSn):
        self.x = np.append(self.x, Sn)
        self.y = np.append(self.y, rhoSn)
        self.yerr = np.append(self.yerr, drhoSn)