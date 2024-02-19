#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:03:47 2024

@author: francesco
"""

import numpy as np
import matplotlib.pyplot as plt

nlds = np.load('nlds.npy', allow_pickle = True)
gsfs = np.load('gsfs.npy', allow_pickle = True)
nldvals_old, nldchis_old, gsfvals_old, gsfchis_old = np.load('../163Dy-alpha-2018/Fittings/data/generated/nld_gsf_vals.npy', allow_pickle = True)
optimal_parameters_iters = 20
energy_bin = 45
rhochis = np.zeros((len(nlds),2))
gsfchis = np.zeros((len(gsfs),2))
for i, nld in enumerate(nlds):
    
    rhochis[i,0] = nld.y[energy_bin]
    rhochis[i,1] = nld.chi2
    
for i, gsf in enumerate(gsfs):
    gsfchis[i,0] = gsf.y[energy_bin]
    gsfchis[i,1] = gsf.chi2
    
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].plot(nldvals_old, nldchis_old, 'y.',alpha=0.4, markersize = 3)
axs[1].plot(gsfvals_old, gsfchis_old, 'y.',alpha=0.1, markersize = 3)
axs[0].plot(rhochis[:,0],rhochis[:,1],'b.',alpha=0.4, markersize = 3)
axs[1].plot(gsfchis[:,0],gsfchis[:,1],'b.',alpha=0.1, markersize = 3)
axs[0].plot(rhochis[:optimal_parameters_iters,0],rhochis[:optimal_parameters_iters,1],'r.',alpha=0.7, markersize = 5)
axs[1].plot(gsfchis[:optimal_parameters_iters,0],gsfchis[:optimal_parameters_iters,1],'r.',alpha=0.7, markersize = 5)

fig.show()


