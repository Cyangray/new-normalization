#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:32:11 2024

@author: francesco
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from readlib import readstrength, readldmodel, search_string_in_file
from systlib import import_ocl, D2rho, chisquared, drho

#paths
talys_path = '/home/francesco/talys/'

#constants. Don't play with these
hc = 197.3269804 #MeV*fm
const = 1/(3*np.pi**2*hc**2*10) #mb

#Don't modify unless you know what you're doing
A = 166
L1min = 7
L2max = 18
NLD_pathstrings = ['FG']
nrhos = 51
blist = np.linspace(0.75,1.25,nrhos)
target_spin = 3.5
Sn = 6.243
a0 = -0.8433
a1 = 0.1221
spin_cutoff_low = 5.546
spin_cutoff_high = 6.926
cutoff_unc = 0.00

#plotting
scaling_factor = 0.9


D0 = 4.35
D0_err = 0.15
Gg_mean = 84
Gg_sigma = 5
Gglist = np.linspace(73,97,25)

rho_Sn_err_up = drho(target_spin, spin_cutoff_high, spin_cutoff_high*cutoff_unc, D0, D0_err)
rho_Sn_err_down = drho(target_spin, spin_cutoff_low, spin_cutoff_low*cutoff_unc, D0, D0_err)
rho_lowerlim = D2rho(D0, target_spin, spin_cutoff_low)
rho_upperlim = D2rho(D0, target_spin, spin_cutoff_high)
rho_mean = (rho_lowerlim - rho_Sn_err_down + rho_upperlim + rho_Sn_err_up)/2
rho_sigma = rho_upperlim + rho_Sn_err_up - rho_mean

database_path = 'Make_dataset/166Ho-database_' + NLD_pathstring + '/'

#load best fits
best_fits_167Ho = np.load('/home/francesco/Documents/164Dy-experiment/Python_normalization/data/generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
best_gsf_167Ho = best_fits_167Ho[1]
best_gsf_167Ho.clean_nans()
#best_gsf_167Ho.delete_point([-5,-4,-3,-2,-1])

best_fits = np.load('data/generated/best_fits_' + NLD_pathstring + '.npy', allow_pickle = True)
best_gsf = best_fits[1]
best_gsf.clean_nans()
best_gsf.delete_point(-1)
best_nld = best_fits[0]
best_nld.clean_nans()
extr_path = best_nld.path[:-10] + 'fermigas.cnt'
extr_mat = import_ocl(extr_path, a0, a1, fermi = True)
extr_vals = []
nld_vals = []
nld_errvals = []
for idx, E in enumerate(best_nld.energies):
    if E > 1.5:
        idx2 = np.argwhere(extr_mat[:,0] == E)[0,0]
        extr_vals.append(extr_mat[idx2,1])
        nld_vals.append(best_nld.y[idx])
        nld_errvals.append(best_nld.yerr[idx])
chisq = chisquared(extr_vals, nld_vals, nld_errvals, DoF=1, method = 'linear',reduced=True)
print(best_gsf.L1)
print(best_gsf.L2)
print(best_gsf.rho)
print(best_gsf.drho)
print(best_gsf.spin_cutoff)
print(best_gsf.Gg)
#load 166Ho GognyM1
pathM1 = talys_path  + 'structure/gamma/gognyM1/Ho.psf'
GognyM1 = np.loadtxt(pathM1, skiprows = search_string_in_file(pathM1, f'A= {A}') + 2, max_rows = 300)

#import experimental nld and gsf
nld_mat = np.genfromtxt('data/generated/nld_' + NLD_pathstring + '_whole.txt', unpack = True).T
gsf_mat = np.genfromtxt('data/generated/gsf_' + NLD_pathstring + '_whole.txt', unpack = True).T
#delete rows with nans
nld_mat = nld_mat[~np.isnan(nld_mat).any(axis=1)]
gsf_mat = gsf_mat[~np.isnan(gsf_mat).any(axis=1)]

#delete some points?
#delete last rows in gsf
gsf_mat = np.delete(gsf_mat,[-1],0)

#import known levels
known_levs = import_ocl('data/rholev.cnt',a0,a1)

#import TALYS calculated GSFs
TALYS_strengths = [readstrength(67, 166, 1, 1, strength, 1) for strength in range(1,9)]

#import TALYS calculated NLDs
TALYS_ldmodels = [readldmodel(67, 166, ld, 1, 1, 1) for ld in range(1,7)]

#start plotting
cmap = matplotlib.cm.get_cmap('YlGnBu')
fig0,ax0 = plt.subplots(figsize = (5.0*scaling_factor, 3.75*scaling_factor), dpi = 300)
fig1,ax1 = plt.subplots(figsize = (5.0*scaling_factor, 3.75*scaling_factor), dpi = 300)
ax0.plot(np.zeros(1), np.zeros([1,5]), color='w', alpha=0, label=' ')
singleaxs = [ax0,ax1]
chi2_lim = [9,13]
chi2_lim_energies = [known_levs[int(chi2_lim[0]),0], known_levs[int(chi2_lim[1]),0]]
ax0.axvspan(chi2_lim_energies[0], chi2_lim_energies[1], alpha=0.2, color='red',label='Fitting intv.')
ax0.plot(known_levs[:,0],known_levs[:,1],'k-',label='Known lvs.')

#############################################


#import experimental nld and gsf
nld_mat_n = np.genfromtxt('/home/francesco/Documents/newnorm/data/generated/nld_whole.txt', unpack = True).T
gsf_mat_n = np.genfromtxt('/home/francesco/Documents/newnorm/data/generated/gsf_whole.txt', unpack = True).T
#delete rows with nans
nld_mat_n = nld_mat_n[~np.isnan(nld_mat_n).any(axis=1)]
gsf_mat_n = gsf_mat_n[~np.isnan(gsf_mat_n).any(axis=1)]

#delete some points?
#delete last rows in gsf
gsf_mat_n = np.delete(gsf_mat_n,[-1],0)


bf_nld, bf_gsf = np.load('/home/francesco/Documents/newnorm/data/generated/best_fits.npy', allow_pickle = True)
for ax, val_matrix_n in zip(singleaxs, [nld_mat_n, gsf_mat_n]):
    #ax.fill_between(val_matrix_n[:,0], val_matrix_n[:,2], val_matrix_n[:,-2], color = 'k', alpha = 0.2, label=r'2$\sigma$ conf.')
    ax.fill_between(val_matrix_n[:,0], val_matrix_n[:,2], val_matrix_n[:,3], color = 'y', alpha = 0.2, label=r'1$\sigma$ conf._n')
    ax.errorbar(val_matrix_n[:,0], val_matrix_n[:,1],yerr=val_matrix_n[:,4], fmt = '.', color = 'k', ecolor='k', label='This work_n')




#ax0.errorbar(bf_nld.energies, bf_nld.y, yerr = bf_nld.yerr, fmt = '.', color = 'k', ecolor='k')
#ax1.errorbar(bf_gsf.energies, bf_gsf.y, yerr = bf_gsf.yerr, fmt = '.', color = 'k', ecolor='k')
###############################################

ax0.errorbar(Sn, rho_mean,yerr=rho_sigma,ecolor='g',linestyle=None, elinewidth = 4, capsize = 5, label=r'$\rho$ at Sn')

#Plot experiment data
for ax, val_matrix in zip(singleaxs, [nld_mat, gsf_mat]):
    #ax.fill_between(val_matrix[:,0], val_matrix[:,2], val_matrix[:,-2], color = 'c', alpha = 0.2, label=r'2$\sigma$ conf.')
    ax.fill_between(val_matrix[:,0], val_matrix[:,3], val_matrix[:,-3], color = 'b', alpha = 0.2, label=r'1$\sigma$ conf.')
    ax.errorbar(val_matrix[:,0], val_matrix[:,1],yerr=val_matrix[:,-1], fmt = '.', color = 'b', ecolor='b', label='This work')
    ax.set_yscale('log')
ax0.set_xlabel(r'$E_x$ [MeV]')
ax1.set_xlabel(r'$E_\gamma$ [MeV]')
#plot TALYS strengths

stls = ['-','--','-.',':','-','--','-.',':']
for i, TALYS_strength in enumerate(TALYS_strengths):
    if i<4:
        col = 3
    else:
        col = 8
    ax1.plot(TALYS_strength[:,0],TALYS_strength[:,1] + TALYS_strength[:,2], color = cmap(col/9), linestyle = stls[i], alpha = 0.8, label = 'strength %d'%(i+1))

#plot TALYS nld
for i, TALYS_ldmodel in enumerate(TALYS_ldmodels):
    if i<3:
        col = 3
    else:
        col = 5
    ax0.plot(TALYS_ldmodel[:,0],TALYS_ldmodel[:,1], color = cmap(col/6), linestyle = stls[i], alpha = 0.8, label = 'ldmodel %d'%(i+1))
ax0.plot(extr_mat[9:,0], extr_mat[9:,1], color = 'k', linestyle = '--', alpha = 1, label = NLD_pathstring + ' extrap.')

ax1.errorbar(best_gsf_167Ho.energies, best_gsf_167Ho.y, yerr = best_gsf_167Ho.yerr, label = r'$^{167}$Ho vanilla')







#plot
ranges = [[1,7], [np.log10(1e-8), np.log10(4e-7)]]
ax0.set_ylabel(r'NLD [MeV$^{-1}$]')
ax1.set_ylabel(r'GSF [MeV$^{-3}$]')
ax0.set_xlim(-0.8,7.5)
ax0.set_ylim(1e1,1e7)
ax1.set_xlim(0.2,6.2)
ax1.set_ylim(8e-9,4e-7)

ax0.legend(loc = 'lower right', ncol = 2, frameon = False)
ax1.legend(loc = 'upper left', ncol = 2, frameon = False)
fig0.tight_layout()
fig1.tight_layout()
fig0.show()
fig1.show()
fig0.savefig('pictures/nld'+NLD_pathstring+'.png', format = 'png')
fig1.savefig('pictures/gsf'+NLD_pathstring+'.png', format = 'png')
fig0.savefig('pictures/nld'+NLD_pathstring+'.pdf', format = 'pdf')
fig1.savefig('pictures/gsf'+NLD_pathstring+'.pdf', format = 'pdf')