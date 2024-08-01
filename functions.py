#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:27:09 2024

@author: francesco
"""

import numpy as np
from dicts_and_consts import const
from scipy.interpolate import interp1d
from utils import Z2Name, search_string_in_file

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

def calc_errors_chis(lst, lab, return_best_fit = False):
    xx = lst[4].x #all energy or temperature vectors are alike. Pick the 5th, but any would do
    val_matrix = np.zeros((xx.size,5))
    
    #find nld or gsf with least chi2-score
    lstargmin = np.nanargmin([el.chi2 for el in lst])
    lstmin = lst[lstargmin]
    
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
            if hasattr(lst[4], 'yerr'):
                row[:] = [x, lstmin.y[i], errmin, errmax, lstmin.yerr[i]]
            else:
                row[:] = [x, lstmin.y[i], errmin, errmax, np.nan]
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

def clean_valmatrix(val_matrix):
    return val_matrix[~np.isnan(val_matrix).any(axis=1)]

def make_TALYS_tab_file(talys_nld_path, talys_nld_cnt, A, Z):
    '''
    Function that incorporates the talys_nld_cnt.txt produced by counting, into
    the big Zz.tab file from TALYS.
    '''
    fmt = "%7.2f %6.3f %9.2E %8.2E %8.2E" + 30*" %8.2E"
    newfile_content = ''
    isotope_strip = f'{Z2Name(Z)}{A}'  #f'Z={Z: >3} A={A: >3}'
    isotopeline = 100000
    with open(talys_nld_path, 'r') as read_obj:
        for n, line in enumerate(read_obj):
            stripped_line = line
            new_line = stripped_line
            if isotope_strip in line:
                isotopeline = n
            if (n >= isotopeline + 3) and ((n - isotopeline + 3) < len(talys_nld_cnt)):
                row = talys_nld_cnt[n - isotopeline - 3]
                new_line_string = fmt % tuple(row) + '\n'
                new_line = new_line_string
            newfile_content += new_line
    
    with open(talys_nld_path, 'w') as write_obj:
        write_obj.write(newfile_content)
        
def make_scaled_talys_nld_cnt(nld_obj):
    '''
    Function that creates two new .tab files by scaling up and down the experimental
    values of the nld according to the statistical errors.
    '''
    
    #ocl_nld_path = nld_obj.path[:-10] + 'talys_nld_cnt.txt'
    #talys_nld_cnt = nld_obj.talys_nld_cnt#np.loadtxt(ocl_nld_path)
    rel_errs = nld_obj.yerr/nld_obj.y
    max_Ex = nld_obj.x[-1]
    #Sn_rel_err = nld_obj.drho/nld_obj.rho
    tab_energies = nld_obj.talys_nld_cnt[:,0]
    rel_errs_tab = np.zeros_like(tab_energies)
    talys_nld_cnt_up = nld_obj.talys_nld_cnt.copy()
    talys_nld_cnt_down = nld_obj.talys_nld_cnt.copy()
    for i, Ex in enumerate(tab_energies):
        if Ex <= max_Ex:
            #for energies less than Ex_max: interpolate, find relative error
            rel_errs_tab[i] = np.interp(Ex, nld_obj.x, rel_errs)
        else:
            #for energies above Sn: as for Sn
            rel_errs_tab[i] = rel_errs[-1]
        talys_nld_cnt_up[i, 2:] = nld_obj.talys_nld_cnt[i, 2:]*(1.+rel_errs_tab[i])
        talys_nld_cnt_down[i, 2:] = nld_obj.talys_nld_cnt[i, 2:]*(1.-rel_errs_tab[i])
        
    return [talys_nld_cnt_down, talys_nld_cnt_up]
        
def make_E1_M1_files_core(gsf, A, Z, M1, target_folder, high_energy_interp, units):
    '''
    Function that takes the energies and the values of gsf and writes two tables 
    for both E1 and M1 ready to be taken as input by TALYS.
    '''
    
    if high_energy_interp is not None:
        gsf = np.vstack((gsf,high_energy_interp))
    
    gsf_folder_path = ''
    if target_folder is not None:
        if target_folder != '':
            if target_folder[-1] != '/':
                target_folder = target_folder + '/'
        gsf_folder_path = target_folder
    
    fn_gsf_outE1 = gsf_folder_path + "gsfE1.dat"
    fn_gsf_outM1 = gsf_folder_path + "gsfM1.dat"
    
    # The file is/should be writen in [MeV] [MeV^-3] [MeV^-3]
    if gsf[0, 0] == 0:
        gsf = gsf[1:, :]
    Egsf = gsf[:, 0]
    
    if isinstance(M1,float):
        method = 'frac'
    elif isinstance(M1, list):
        if len(M1) == 3:
            method = 'SLO'
        elif len(M1) == 6:
            method = 'SLO2'
    
    if method == 'frac':
        gsfE1 = gsf[:, 1]*(1-M1)
        gsfM1 = gsf[:, 1]*M1
    elif method =='SLO':
        M1_vals = SLO_arglist(gsf[:,0], M1[:3])
        gsfE1 = gsf[:,1] - M1_vals
        gsfM1 = M1_vals
    elif method =='SLO2':
        M1_vals1 = SLO_arglist(gsf[:,0], M1[:3])
        M1_vals2 = SLO_arglist(gsf[:,0], M1[3:])
        M1_vals = M1_vals1 + M1_vals2
        gsfE1 = gsf[:,1] - M1_vals
        gsfM1 = M1_vals

    # REMEMBER that the TALYS functions are given in mb/MeV (Goriely's tables)
    # so we must convert it (simple factor)
    if units == 'mb':
        factor_from_mb = 8.6737E-08   # const. factor in mb^(-1) MeV^(-2)
    else:
        factor_from_mb = 1.0
    
    fE1 = log_interp1d(Egsf, gsfE1, fill_value="extrapolate")
    fM1 = log_interp1d(Egsf, gsfM1, fill_value="extrapolate")
    
    Egsf_out = np.arange(0.1, 30.1, 0.1)
    
    headerE1 = f" Z =  {Z} A =  {A}\n" + "  U[MeV]  fE1[mb/MeV]"
    headerM1 = f" Z =  {Z} A =  {A}\n" + "  U[MeV]  fM1[mb/MeV]"
    # gsfE1 /= factor_from_mb
    np.savetxt(fn_gsf_outE1, np.c_[Egsf_out, fE1(Egsf_out)/factor_from_mb],
               fmt="%9.3f%12.3E", header=headerE1)
    # gsfM1 /= factor_from_mb
    np.savetxt(fn_gsf_outM1, np.c_[Egsf_out, fM1(Egsf_out)/factor_from_mb],
               fmt="%9.3f%12.3E", header=headerM1)
    return Egsf_out, fE1(Egsf_out)/factor_from_mb, fM1(Egsf_out)/factor_from_mb

def make_E1_M1_files_simple(energies, values, A, Z, M1 = 0.1, target_folder = None, high_energy_interp=None, delete_points = None, units = 'mb'):
    '''
    Function that takes the energies and the values of gsf and writes two tables 
    for both E1 and M1 ready to be taken as input by TALYS.
    '''
    gsf = np.c_[energies,values]
    if delete_points is not None:
        gsf = np.delete(gsf, delete_points, 0)
    return make_E1_M1_files_core(gsf, A, Z, M1, target_folder, high_energy_interp, units = units)

def SLO_arglist(E, args):
    '''
    Standard Lorentzian, adapted from Kopecky & Uhl (1989) eq. (2.1)
    '''
    E0, Gamma0, sigma0 = args
    funct = const * sigma0 * E * Gamma0**2 / ( (E**2 - E0**2)**2 + E**2 * Gamma0**2 )
    return funct

def GLO_arglist(E, args):
    '''
    General Lorentzian, adapted from Kopecky & Uhl (1989) eq. (2.4)
    '''
    T, E0, Gamma0, sigma0 = args
    Gamma = Gamma0 * (E**2 + 4* np.pi**2 * T**2) / E0**2
    param1 = (E*Gamma)/( (E**2 - E0**2)**2 + E**2 * Gamma**2 )
    param2 = 0.7*Gamma0*4*np.pi**2 *T**2 /E0**5
    funct = const * (param1 + param2)*sigma0*Gamma0
    return funct

def GLO_hybrid_arglist(E, args):
    '''
    Goriely's Hybrid model
    Coded from the fstrength.f subroutine in TALYS
    '''
    T, E0, Gamma0, sigma0 = args
    ggredep = 0.7 * Gamma0 * ( E/E0 + (2*np.pi*T)**2/(E*E0))
    enumerator = ggredep*E
    denominator = (E**2 - E0**2)**2 + E**2 * ggredep * Gamma0
    factor1 = enumerator/denominator
    return const*sigma0*Gamma0*factor1

def log_interp1d(xx, yy, **kwargs):
    """ Interpolate a 1-D function.logarithmically """
    logy = np.log(yy)
    lin_interp = interp1d(xx, logy, kind='linear', **kwargs)
    log_interp = lambda zz: np.exp(lin_interp(zz))
    return log_interp

def readldmodel_path(path, A, Z):
    '''
    reads the nld model from the output file of TALYS, given its path.
    '''
    
    max_rows = 55
    Zstring = str(Z).rjust(3)
    Nstring = str(A-Z).rjust(3)
    start = search_string_in_file(path, f'Level density parameters for Z={Zstring} N={Nstring}')

    Parity = search_string_in_file(path, 'Positive parity', skip_rows = start)
    if Parity:
        posneg = True
        rowsskip = Parity + 2
        rowsskip2 = search_string_in_file(path, 'Negative parity', skip_rows = start) + 2
    else:
        posneg = False
        rowsskip = search_string_in_file(path, '    Ex     a    sigma', skip_rows = start) + 2
    
    if posneg:
        pos_par = np.loadtxt(path, skiprows = rowsskip, max_rows = max_rows)
        neg_par = np.loadtxt(path, skiprows = rowsskip2, max_rows = max_rows)
        out_matrix1 = np.zeros(pos_par.shape)
        out_matrix1[:,0] = pos_par[:,0]
        out_matrix1[:,1:] = np.add(pos_par[:,1:], neg_par[:,1:])
    else:
        out_matrix1 = np.loadtxt(path, skiprows = rowsskip, max_rows = max_rows)
    
    out_matrix2 = out_matrix1.copy()
    if out_matrix1.shape[1] == 11:
        out_matrix2 = np.zeros((out_matrix1.shape[0], out_matrix1.shape[1]+2))
        out_matrix2[:,0] = out_matrix1[:,0]
        out_matrix2[:,3:] = out_matrix1[:,1:]
        
    return out_matrix2

def readstrength_path(path):
    #read the strength function from the output file
    #if path to output file is not given, it will be inferred from the simulation parameters
    rowsskip = search_string_in_file(path, 'f(E1)') + 2
    return np.loadtxt(path, skiprows = rowsskip, max_rows = 73)