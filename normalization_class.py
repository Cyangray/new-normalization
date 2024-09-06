#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Normalization class, used for the normalization procedure of the Oslo method,
it uses Monte Carlo simulations to propagate the systematic errors from the input
data to the final results, for customized error PDFs. This is a systematized, 
effectivized and expanded version of the algorithm used in https://doi.org/10.1103/PhysRevC.107.034605
'''

import time
import numpy as np
from functions import D2rho, drho, chisquared, import_Anorm_alpha, import_Bnorm, calc_errors_chis, clean_valmatrix, make_TALYS_tab_file, make_E1_M1_files_simple, make_scaled_talys_nld_cnt, readldmodel_path, readstrength
import matplotlib
import matplotlib.pyplot as plt
from subprocess import call
import os
from scipy.special import erf
from scipy.integrate import quad
from nld_gsf_classes import import_ocl, nld, gsf, nld_p, gsf_p, ncrate, MACS
rng = np.random.default_rng()
import shutil
from utils import Z2Name
from multiprocess import Pool
import tqdm
import copy

#working directory
root_folder = os.getcwd()

#some useful lists
supported_TALYS_versions = ['1.96', '2.00']
NLD_keys = ['D0', 'L1', 'L2', 'H1', 'H2', 'TL1', 'TL2', 'TH1', 'TH2', 'extr_model', 'Ex_low', 's_low', 'sigma', 'FWHM', 'a', 'E1', 'T', 'E0']
couples = [['L1', 'L2'],
           ['H1', 'H2'],
           ['TL1', 'TL2'],
           ['TH1', 'TH2']]

#normalization class
class normalization:
    def __init__(self, rsc_folderpath, oms_path, A, Z, Sn):
        self.rsc_folderpath = rsc_folderpath
        self.A = A
        self.Z = Z
        self.Sn = Sn
        self.counting_code_path = oms_path + '/prog/counting'
        self.normalization_code_path = oms_path + '/prog/normalization'
        
        #read some important data from rhosigchi output
        rhosp = np.genfromtxt(self.rsc_folderpath + '/rhosp.rsg', skip_header=6, max_rows = 1, delimiter =',')
        self.a0 = rhosp[1]/1000
        self.a1 = rhosp[2]/1000
        self.FGmax = int(100/self.a1)
        self.NchBn = 1 + int((self.Sn - self.a0)/self.a1 + 0.5)
        rhopaw_path = self.rsc_folderpath + '/rhopaw.rsg'
        self.rhopaw = import_ocl(rhopaw_path, self.a0, self.a1)
        sigpaw_path = self.rsc_folderpath + '/sigpaw.rsg'
        self.sigpaw = import_ocl(sigpaw_path, self.a0, self.a1)
        self.rhopaw_length = len(self.rhopaw[:,0])
        self.sigpaw_length = len(self.sigpaw[:,0])
        self.dim = min(self.rhopaw_length, self.sigpaw_length)
        self.TALYS_models = False
        self.TALYS_ver = None
        self.start_spin = 0.5
        if self.A % 2 == 0:
            self.start_spin = 0
            
        
    def to_energy(self,index):
        #transforms index to energy
        return self.a0 + self.a1*index
    
    def to_bin(self, E):
        #transforms energy to index
        return round((E - self.a0)/self.a1)
    
    def set_attributes(self, **kwargs):
        '''
        attributes that are useful are
        sigma: spin-cutoff parameter. It can be one value, or a tuple/list in case one wants to e.g. take in consideration the results from both the FG and RMI spin-cutoff formulas
        dsigma: uncertainty in the spin-cutoff parameter. Float or tuple/list, must coincide with sigma. If list, the first value is the error of the first provided value in sigma
        
        D0: average level spacing at Sn (eV). It can be one value, or a tuple/list in case one wants to e.g. take in consideration both the Mughabghab and the RIPL values
        dD0: uncertainty in the average level spacing at Sn  (eV). Float or tuple/list, must coincide with D0. If list, the first value is the error of the first provided value in D0
        
        target_spin: spin in the N-1 nucleus
        
        Gg: average radiative strength (meV)
        dGg: uncertainty in average radiative strength (meV)
        
        Ex_low: the energy where the spin-cutoff value for low excitation energy is evaluated (used for the linear fit, ALEX method in counting.c. see https://doi.org/10.1140/epja/i2015-15170-4 (or the application in https://doi.org/10.1103/PhysRevC.107.034605))
        s_low: the corresponding spin-cutoff parameter at Ex_low
        
        NLD FITTING REGIONS
         - choose whether to give the lower and upper limit for the lower excitation energy fitting interval in energies (ExL1, ExL2), or bin numbers (L1, L2)
        ExL1: lower limit for the low excitation energy interval where the NLD is fitted (energy (MeV), float)
        ExL2: upper limit for the low excitation energy interval where the NLD is fitted (energy (MeV), float)
        or 
        L1: lower limit for the low excitation energy interval where the NLD is fitted (bin, int)
        L2: upper limit for the low excitation energy interval where the NLD is fitted (bin, int)
        
         - choose whether to give the lower and upper limit for the higher excitation energy fitting interval in energies (ExH1, ExH2), or bin numbers (H1, H2)
        ExH1: lower limit for the high excitation energy interval where the NLD model is fitted (energy (MeV), float)
        ExH2: upper limit for the high excitation energy interval where the NLD model is fitted (energy (MeV), float)
        or 
        H1: lower limit for the high excitation energy interval where the NLD model is fitted (bin, int)
        H2: upper limit for the high excitation energy interval where the NLD model is fitted (bin, int)
        
        TRANSMISSION COEFFICIENT FITTING REGIONS
        - choose whether to give the lower and upper limit for the lower gamma energy fitting interval for the transmission coefficient in energies (EgL1, EgL2), or bin numbers (TL1, TL2)
        EgL1: lower limit for the low gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        EgL2: upper limit for the low gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        or 
        TL1: lower limit for the low gamma energy interval where the NLD is fitted (bin, int)
        TL2: upper limit for the low gamma energy interval where the NLD is fitted (bin, int)
       
        - choose whether to give the lower and upper limit for the higher gamma energy fitting interval for the transmission coefficient in energies (EgH1, EgH2), or bin numbers (TH1, TH2)
        EgH1: lower limit for the high gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        EgH2: upper limit for the high gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        or 
        TH1: lower limit for the high gamma energy interval where the transmission coefficient is fitted (bin, int)
        TH2: upper limit for the high gamma energy interval where the transmission coefficient is fitted (bin, int)
        
        extr_model: NLD model to extrapolate the data to Sn. Either 1 (constant temperature) or 2 (FG)
        
        a, E1, T, E0: CT or FG parameters
        '''
        
        self.a = 1.0
        self.T = 1.0
        self.E0 = 0.0
        self.E1 = 0.0
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.rho_variables = ['sigma', 'D0']
        self.rho_dvariables = ['d' + variable for variable in self.rho_variables]
        
        for variable, dvariable in zip(self.rho_variables, self.rho_dvariables):
            var = getattr(self, variable)
            dvar = getattr(self, dvariable)
            assert type(var) == type(dvar)
            
            setattr(self, variable + 'double', False)
            if isinstance(var, list):
                setattr(self, variable + 'double', True)
                if var[0] > var[1]:
                    lowest = 1
                    highest = 0
                else:
                    lowest = 0
                    highest = 1
                
                setattr(self, variable + 'max', var[highest])            
                setattr(self, variable + 'min', var[lowest])
                setattr(self, 'd' + variable + 'max', dvar[highest])
                setattr(self, 'd' + variable + 'min', dvar[lowest])
                setattr(self, variable, sum(var)/2)
                setattr(self, dvariable, np.sqrt(dvar[0]**2 + dvar[1]**2)/2)
                  
        #If L1, L2... given as energies, translate them into bins, and print something
        for i, function in enumerate(['x', 'g']):
            for j, LH in enumerate(['H', 'L']):
                prefix = ''
                if function == 'g':
                    prefix = 'T'
                
                if ((prefix + LH + '1') in kwargs or ('E' + function + LH + '1') in kwargs) and ((prefix + LH + '2') in kwargs or ('E' + function + LH + '2') in kwargs):

                    for lowhigh in ['1', '2']:
                        index = prefix + LH + lowhigh
                        energy = 'E' + function + LH + lowhigh
                        if (index not in kwargs) and (energy in kwargs):
                            setattr(self, index, self.to_bin(getattr(self, energy)))
                            print(f'{index} energy {getattr(self, energy)} rounded to {self.to_energy(getattr(self, index))} to coincide with bin number {getattr(self, index)}')
                        elif (index in kwargs) and (energy not in kwargs):
                            setattr(self, energy, self.to_energy(getattr(self, index)))
                        elif (index in kwargs) and (energy in kwargs):
                            raise Exception(f'Please provide only {index} or {energy}')
                else: #If L1, L2... or their energies not given, guess values
                    index = prefix + LH
                    energy = 'E' + function + LH
                    index_low_guess, index_high_guess = self.guess_indexes(function + LH)
                    print(f'Values for {index + "1"} and {index + "2"} not given. Set automatically to {index_low_guess} (= {self.to_energy(index_low_guess)} MeV) and {index_high_guess} (= {self.to_energy(index_high_guess)} MeV).')
                    setattr(self, index + '1', index_low_guess)
                    setattr(self, index + '2', index_high_guess)
                    setattr(self, energy + '1', self.to_energy(index_low_guess))
                    setattr(self, energy + '2', self.to_energy(index_high_guess))
                    
    def guess_indexes(self, label):
        '''
        Function guessing L1, L2... values using the same algorithms as in counting.c
        '''
        if label == 'xH':
            for i in range(self.dim-1, 5, -1):
                if self.rhopaw[i,1] > 0.010 and self.rhopaw[i,2] > 0.010 and self.rhopaw[i,1] > 1.2*self.rhopaw[i,2]:
                    return [i - int(i/6.0+0.5), i]
        
        elif label == 'xL':
            self.import_low_Ex_nld_raw()
            i1 = int((0.500 - self.a0)/self.a1 + 0.5)
            i2 = int((1.000/self.a1) + 0.5)
            if i1 < 0:
                i1 = 1
            if i2 < 2:
                i2 = 2
            i0 = i1
            for i in range(i1, i1 + i2, 1):
                if self.nld_lvl_raw[i,1] <= 0:
                    i0 = i
            L1 = i0 + 1
            L2 = L1 + i2
            if L2 > self.H1 - 2:
                L2 = self.H1 - 2
            return [L1, L2]
        
        elif label == 'gH':
            for i in range(self.dim-1, self.dim//2, -1):
                if self.sigpaw[i,1] > 0. and self.sigpaw[i,2] > 0. and self.sigpaw[i,1] > 1.2*self.sigpaw[i,2]:
                    return [i - int(i/6. + 0.5), i]
                    
        elif label == 'gL':
            for i in range(0, self.dim//2, 1):
                if self.sigpaw[i,1] > 0. and self.sigpaw[i,2] > 0. and self.sigpaw[i,1] > 1.2*self.sigpaw[i,2]:
                    i1 = i
                    break
            if i1 > self.dim//2:
                i1 = self.dim//2
            i2 = i1 + int(1.000/self.a1 + 0.5)
            if i1 < 0:
                i1 = 1
            if i2 < 2:
                i2 = 2
            i0 = i1
            for i in range(i1, i1 + i2, 1):
                if self.sigpaw[i,1] <= 0:
                    i0 = i
            TL1 = i0 + 1
            TL2 = TL1 + (i2 - i1)
            if TL2 > self.TH1 - 2:
                TL2 = self.TH1 - 2
            return [TL1, TL2]
            
    def set_TALYS_version(self, talys_root_path, talys_executable_path, TALYS_ver):
        # TALYS_ver: either '1.96' or '2.00'
        self.talys_root_path = talys_root_path
        self.TALYS_ver = TALYS_ver
        self.talys_executable_path = talys_executable_path
    
    def set_variation_intervals(self, std_devs, **kwargs):
        
        '''
        If you provide a variable in the kwargs here, it will be added to the 
        free parameters and will be varied in the MC simulation. Otherwise the
        variable will be kept fixed.
        
        std_devs: how many standard deviations the parameter will be varied in the MC simulation.
        
        kwargs are in the form
        'D0': [min, max],
        'L1': [min, max]...
        
        for extr_model, only 1 and 2 are possible anyway. If you want to pick randomly between them, put
        'extr_model: 'yes' (or any non-empty string, actually)
        
        numerical values for min, max
        if min or max = 'err', the error value provided in set_attributes will be used
        
        if you want L1 and L2 to take all possible values within a range, give them the same limits (same for H1,H2,TL1...). L2 always bigger than L1.
        
        sigmaflat (boolean): if True, the probability of the spin-cutoff parameter to be between the two sigmas provided is the same as for getting one of the two sigmas. (if a normal error distribution is a Gaussian, with flat = True the distribution is Gaussian outside the interval between the sigmas, and flat within.)
        D0flat (boolean): same as sigmaflat, just for D0.
        '''
        
        self.std_devs = std_devs
        self.free_parameters = []
        self.n_free_parameters = 0
        
        for variable in self.rho_variables:
            setattr(self, variable + 'flat', False)
        
        for key, value in kwargs.items():
            self.free_parameters.append(key)
            if 'flat' in key:
                setattr(self, key, value)
            elif ((key == 'sigma') and self.sigmadouble) or ((key == 'D0') and self.D0double):
                self.n_free_parameters += 1
                par_range = value
                if par_range[0] == 'err':
                    par_range[0] = getattr(self, key + 'min') - self.std_devs*getattr(self, 'd' + key + 'min')
                if par_range[1] == 'err':
                    par_range[1] = getattr(self, key + 'max') + self.std_devs*getattr(self, 'd' + key + 'max')
                setattr(self, key+'_range', par_range)
            else:
                self.n_free_parameters += 1
                par_range = value
                if par_range[0] == 'err':
                    par_range[0] = getattr(self, key) - self.std_devs*getattr(self, 'd' + key)
                if par_range[1] == 'err':
                    par_range[1] = getattr(self, key) + self.std_devs*getattr(self, 'd' + key)
                setattr(self, key+'_range', par_range)
        
    def rhouncdistr(self, varvalues):
        '''
        Evaluates the probability density function of the considered variables
        '''
        pdf = 1.0
        for x, variable in zip(varvalues, self.rho_variables):
            
            var = getattr(self, variable)
            dvar = getattr(self, 'd' + variable)
            vardouble = getattr(self, variable + 'double')
            varflat = getattr(self, variable + 'flat')
            
            if vardouble:
                varmin = getattr(self, variable + 'min')
                varmax = getattr(self, variable + 'max')
                dvarmin = getattr(self, 'd' + variable + 'min')
                dvarmax = getattr(self, 'd' + variable + 'max')
                
            if vardouble and not varflat:
                stderr = varmax - var + dvarmax
                partial_pdf = np.exp(-np.power((x - var)/stderr, 2.)/2)/(stderr*np.sqrt(2*np.pi))
                    
            elif vardouble and varflat:
                norm_const = (dvarmin + dvarmax)*np.sqrt(np.pi/2) + varmax - varmin
                if x < varmin:
                    partial_pdf = np.exp(-np.power((x - varmin)/dvarmin, 2.)/2)/norm_const
                elif x < varmax:
                    partial_pdf = 1.0/norm_const
                else:
                    partial_pdf = np.exp(-np.power((x - varmax)/dvarmax, 2.)/2)/norm_const
                
            else:
                partial_pdf = np.exp(-np.power((x - var)/dvar, 2.)/2)/(dvar*np.sqrt(2*np.pi))
            
            pdf = pdf * partial_pdf
            
        return pdf
    
    def rho_chi2score(self, varvalues):
        '''
        translates the gaussian or flat-gaussian uncertainty distribution into a chi2 score.
        '''
        return -2*np.log(self.rhouncdistr(varvalues))
    
    def low_levels_chi2(self, rholev, curr_nld):
        '''
        Calculate the chi2 score from the NLD fit to the level density calculated
        from the known levels
        '''
        exp = curr_nld.y[((curr_nld.x >= self.to_energy(self.L1)) & (curr_nld.x <= self.to_energy(self.L2)))]
        err = curr_nld.yerr[(curr_nld.x >= self.to_energy(self.L1)) & (curr_nld.x <= self.to_energy(self.L2))]
        theo = rholev[(rholev[:,0] >= self.to_energy(self.L1)) & (rholev[:,0] <= self.to_energy(self.L2))][:,1]
        chi2_score = chisquared(theo, exp, err, DoF=1, method = 'linear', reduced=False)
        return chi2_score
    
    def dvar_interp(self, xn, variable):
        '''
        Calculate the interpolated variable uncertainty if it falls between two known values
        '''
        varmin = getattr(self, variable + 'min')
        varmax = getattr(self, variable + 'max')
        dvarmin = getattr(self, 'd' + variable + 'min')
        dvarmax = getattr(self, 'd' + variable + 'max')
        
        x = varmax - varmin
        y = dvarmax - dvarmin
        if y==0:
            return dvarmin
        a = y/x
        yn = xn*a
        return dvarmin + yn
        
    
    def import_low_Ex_nld_from_file(self, path):
        '''
        If you have a rholev.cnt, you can give it to this method and it will read the low Ex level density
        '''
        self.nld_lvl = import_ocl(path, self.a0, self.a1, no_errcol=True)
    
    def NLD_from_countingdat(self, all_levels, Ex, binsize):
        
        ld = 0
        bin_lowlim = Ex - binsize/2
        bin_highlim = Ex + binsize/2
        for el in all_levels:
            if bin_lowlim <= el < bin_highlim:
                ld += 1
        
        ans = nld/abs(self.a1)
        
        return ans 
    
    def import_low_Ex_nld_raw(self, countingdat_path = None, binsize = 1.0):
        '''
        This will use the first (naive) algorithm from counting.c to read the low Ex
        level density from counting.dat. This is only used to guess L1 and L2.
        '''
        
        if countingdat_path == None:
            countingdat_path = self.rsc_folderpath + '/counting.dat'
        all_levels = np.loadtxt(countingdat_path)/1000
        self.nld_lvl_raw = np.zeros((self.dim, 2))
        
        for i in range(self.dim):
            self.nld_lvl_raw[i,0] = self.a0 + self.a1*i
            self.nld_lvl_raw[i,1] = self.NLD_from_countingdat(all_levels, self.nld_lvl_raw[i,0], binsize)
        
    def import_low_Ex_nld(self, countingdat_path = None, binsize = 1.0):
        '''
        This will use the readsmooth algorithm from counting.c to read the low Ex
        level density from counting.dat
        '''
        if countingdat_path == None:
            countingdat_path = self.rsc_folderpath + '/counting.dat'
        
        all_levels = np.loadtxt(countingdat_path)/1000
        self.nld_lvl = np.zeros((self.dim, 2))
        
        if self.FWHM > 0:
            sigma = self.FWHM/(2*np.sqrt(2*np.log(2)))/1000
            
            for i in range(self.dim):
                self.nld_lvl[i,0] = self.a0 + self.a1*i
            
            for l, el in enumerate(all_levels):
                for j in range(self.dim):
                    emin = self.a0 + (j-binsize/2)*self.a1
                    emax = self.a0 + (j+binsize/2)*self.a1
                    
                    w0 = erf( (el - emax)/(np.sqrt(2)*sigma) )
                    w1 = erf( (el - emin)/(np.sqrt(2)*sigma) )
                    weight = 0.5*(w1 - w0)/(self.a1)#*1e-3)
                    self.nld_lvl[j,1] += weight
        else:
            self.import_low_Ex_nld_raw(countingdat_path = countingdat_path, binsize = binsize)
            self.nld_lvl = self.nld_lvl_raw
    
    def write_input_cnt(self, rho, drho, D0, L1, L2, H1, H2, TL1, TL2, TH1, TH2, extr_model, Ex_low, s_low, sigma, FWHM, a, E1, T, E0): #TODO: obsolete
        '''
        Writes the counting.c input file
        '''
        lines = []
        lines.append(['{:.6f}'.format(self.A), '1.847000', '{:.6f}'.format(self.Sn), '{:.6f}'.format(rho), '{:.6f}'.format(drho)])
        lines.append([str(int(L1)), str(int(L2)), str(int(H1)), str(int(H2))])
        lines.append([str(int(TL1)), str(int(TL2)), str(int(TH1)), str(int(TH2))])
        lines.append(['5', '{:.6f}'.format(a), '{:.6f}'.format(E1)])
        lines.append(['2', '{:.6f}'.format(T), '{:.6f}'.format(E0)])
        lines.append([str(int(extr_model))])
        lines.append(['0', '-1000.000000', '-1000.000000'])
        lines.append(['0', '-1000.000000', '-1000.000000'])
        lines.append(['1.000000'])
        lines.append(['{:.6f}'.format(Ex_low), '{:.6f}'.format(s_low), '{:.6f}'.format(self.Sn), '{:.6f}'.format(sigma)])
        lines.append(['{:.6f}'.format(FWHM)])
        
        newinput_cnt = ''
        for line in lines:
            curr_line = ' '
            for el in line:
                curr_line += (el + ' ')
            curr_line += '\n'
            newinput_cnt += curr_line
        
        with open('input.cnt', 'w') as write_obj:
            write_obj.write(newinput_cnt)
        
    def run_NLD_sim(self, inputlist):
        '''
        runs the counting.c program, and reads and save the results in a nld object,
        and returns the instance
        '''
        
        vardict = dict(zip(NLD_keys, inputlist))
        for variable in self.rho_variables:
            vardouble = getattr(self, variable + 'double')
            if vardouble:
                vardict['d' + variable] = self.dvar_interp(vardict[variable], variable)
            else:
                vardict['d' + variable] = getattr(self, 'd' + variable)
                
        current_rho = D2rho(vardict['D0'], self.target_spin, vardict['sigma'])
        current_drho = drho(self.target_spin, vardict['sigma'], vardict['dsigma'], vardict['D0'], vardict['dD0'], rho = current_rho)
        D0, L1, L2, H1, H2, TL1, TL2, TH1, TH2, extr_model, Ex_low, s_low, sigma, FWHM, a, E1, T, E0 = inputlist
        
        #TODO: evaluate E0 automatically, as done in counting, or should it be user defined?
        #TODO: same for T
        
        b2 = (sigma**2 - s_low**2)/(self.Sn - Ex_low)
        b1 = s_low**2 - b2*Ex_low;
        
        _, rhoSn_extrap = self.rhofg(self.Sn, a, T, E1, E0, extr_model, b1, b2)
        eta = current_rho/rhoSn_extrap
        
        Lm = int((L1 + L2)/2)
        Hm = int((H1 + H2)/2)
        c1 = (self.rhopaw[Lm-1, 1] + self.rhopaw[Lm, 1] + self.rhopaw[Lm+1, 1]) / 3.
        c2 = (self.rhopaw[Hm-1, 1] + self.rhopaw[Hm, 1] + self.rhopaw[Hm+1, 1]) / 3.
        rhoL = (self.nld_lvl[Lm-1, 1] + self.nld_lvl[Lm, 1] + self.nld_lvl[Lm+1, 1]) / 3.
        rhoL = max(rhoL, 0.01)
        e1 = self.a0 + self.a1 * Lm
        e2 = self.a0 + self.a1 * Hm
        _, rhox = self.rhofg(e2, a, T, E1, E0, extr_model, b1, b2)
        alpha = (np.log(rhoL) + np.log(c2) - np.log(eta*rhox) - np.log(c1)) / (e1-e2)
        Anorm = np.exp(-alpha*e1) * (rhoL)/c1
        c1 = c1/self.corrL(L1, L2, alpha, Anorm)
        c2 = c2/self.corrH(H1, H2, alpha, Anorm, a, T, E1, E0, extr_model, b1, b2, eta)
        
        _, rhox = self.rhofg(e2, a, T, E1, E0, extr_model, b1, b2)
        alpha = (np.log(rhoL) + np.log(c2) - np.log(eta*rhox) - np.log(c1)) / (e1-e2)
        Anorm = np.exp(-alpha*e1) * (rhoL)/c1
        
        #write extrapolation
        
        energies = [self.a0 + self.a1*i for i in range(self.FGmax)]
        extr_mat = np.c_[energies, [self.rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)[1]*eta for Ex in energies]]
        
        #write rhotmopaw
        rhotmopaw = np.zeros_like(energies)
        for i, Ex in enumerate(energies):
            if i<H1:
                rhotmopaw[i] = Anorm*np.exp(alpha * Ex)*self.rhopaw[i,1]
            else:
                _, rhox = self.rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)
                rhotmopaw[i] = eta*rhox
        
        #write normalized rho
        rho_mat = np.zeros((self.dim, 3))
        for i in range(self.dim):
            Ex = self.a0 + self.a1*i
            rho_mat[i,0] = Ex
            rho_mat[i,1] = Anorm * np.exp(alpha*Ex)*self.rhopaw[i,1]
            rho_mat[i,2] = Anorm * np.exp(alpha*Ex)*self.rhopaw[i,2]
        
        #write sigpaw
        sig_mat = np.zeros((self.dim, 3))
        for i in range(self.dim):
            Eg = self.a0 + self.a1*i
            sig_mat[i,0] = Eg
            sig_mat[i,1] = Anorm * np.exp(alpha*Eg)*self.sigpaw[i,1]
            sig_mat[i,2] = Anorm * np.exp(alpha*Eg)*self.sigpaw[i,2]
        
        #extend sig_mat
        sigdimBn = (self.Sn + 0.5 - self.a0)/self.a1 + 0.5
        sigdimBn = int(max(sigdimBn, self.dim))
        nsigL = self.extendL(TL1, TL2, sigdimBn, sig_mat)
        nsigH = self.extendH(TH1, TH2, self.FGmax, sig_mat)
        sigpawext = np.zeros(int(self.FGmax/5))
        for i in range(int(self.FGmax/5)):
            if i < TL2 + 1:
                sigpawext[i] = nsigL[i,1]
            elif i < TH1:
                sigpawext[i] = sig_mat[i,1]
            else:
                sigpawext[i] = nsigH[i,1]
        
        #write spincut
        spincut = np.zeros(self.dim*3)
        sigma2 = np.zeros(self.FGmax)
        for i in range(max(3*self.dim, self.FGmax)):
            Ex = self.a0 + self.a1*i
            sig2, _ = self.rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)
            if i < 3*self.dim:
                spincut[i] = np.sqrt(sig2)
            if i < self.FGmax:
                sigma2[i] = sig2
        
        
        curr_nld = nld_p(rho_mat)
        curr_nld.clean_nans()
        curr_nld.add_rhoSn(self.Sn, current_rho, current_drho)
        for key,value in vardict.items():
            setattr(curr_nld, key, value)
        curr_nld.Anorm = Anorm
        curr_nld.alpha = alpha
        curr_nld.rho = current_rho
        curr_nld.drho = current_drho
        curr_nld.extr_mat = extr_mat
        curr_nld.rhoSn_chi2 = self.rho_chi2score([vardict[variable] for variable in self.rho_variables])
        curr_nld.chi2 = self.low_levels_chi2(self.nld_lvl, curr_nld) + curr_nld.rhoSn_chi2
        curr_nld.sig_mat = sig_mat
        curr_nld.rhotmopaw = rhotmopaw
        curr_nld.sigpawext = sigpawext
        curr_nld.spincut = spincut
        curr_nld.sigma2 = sigma2
        curr_nld.b1 = b1
        curr_nld.b2 = b2
        curr_nld.extr_model = extr_model
        curr_nld.talys_nld_cnt = self.nld_talys(curr_nld)
        curr_nld.Int = self.normalization_integral(curr_nld)
        curr_nld.eta = eta
        return curr_nld
        
    def nld_talys(self, curr_nld):
        stop_spin = 30   # Spin loop stops after 30 iterations
        n_cum = 0.       # Cumulative number of levels, different bin size on Ex!
        ex_bin1 = 0.25   # 0.25 MeV from Ex=0.25 - 5.00 MeV, i= 0-19
        ex_bin2 = 0.50   # 0.50 MeV from Ex=5.50 - 10.0 MeV, i=20-29
        ex_bin3 = 1.00   # 1.00 MeV from Ex=11.0 - 20.0 MeV, i=30-39
        ex_bin4 = 2.50   # 2.50 MeV from Ex=22.5 - 25.0 MeV, i=40-41
        ex_bin5 = 5.00   # 5.00 MeV from Ex=25.0 - 30.0 MeV, i=41-42
        ex_bin6 = 10.0   # 10.0 MeV from Ex=30.0 - 150. MeV, i=43-54
        
        #make energy array for TALYS:
        energies = np.zeros(56)
        
        energies[0] = ex_bin1
        for i in range(55):
            if i < 19:
                energies[i+1] = energies[i] + ex_bin1
            elif i < 29:
                energies[i+1] = energies[i] + ex_bin2
            elif i < 39:
                energies[i+1] = energies[i] + ex_bin3
            elif i < 41:
                energies[i+1] = energies[i] + ex_bin4
            elif i < 42:
                energies[i+1] = energies[i] + ex_bin5
            elif i < 55:
                energies[i+1] = energies[i] + ex_bin6
        nld_calc = np.zeros(56)
        #temp_calc = np.zeros(56)
        talys_nld_cnt = np.zeros((56,35))
        for i in range(56):
            nld_calc[i] = self.rhoexp(energies[i], curr_nld.rhotmopaw)
            if nld_calc[i] > 1e30:
                break
            if i < 20:
                n_cum += nld_calc[i]*ex_bin1
            elif i < 30:
                n_cum += nld_calc[i]*ex_bin2
            elif i < 40:
                n_cum += nld_calc[i]*ex_bin3
            elif i < 42:
                n_cum += nld_calc[i]*ex_bin4
            elif i < 43:
                n_cum += nld_calc[i]*ex_bin5
            elif i < 55:
                n_cum += nld_calc[i]*ex_bin6
            
            talys_nld_cnt[i,0] = energies[i]
            talys_nld_cnt[i,1] = self.temperature(energies[i], curr_nld.rhotmopaw)
            talys_nld_cnt[i,2] = n_cum
            talys_nld_cnt[i,3] = nld_calc[i]
            talys_nld_cnt[i,4] = nld_calc[i]
            for j in range(stop_spin):
                I = j + self.start_spin
                x = nld_calc[i]*self.spin_distribution(energies[i], I, curr_nld.sigma2)
                talys_nld_cnt[i,4+j] = x
        return talys_nld_cnt
        
    def rhoexp(self, Ex, rhotmopaw):
        for i in range(self.FGmax):
            if ((self.a0 + self.a1*i)/1000) > Ex:
                break
        i2 = i
        i1 = i2-1
        rhox = rhotmopaw[i1] + ((rhotmopaw[i2]-rhotmopaw[i1])/self.a1)*(Ex-(self.a0 + self.a1*i1))
        return rhox
    
    def temperature(self, Ex, rhotmopaw):
        for i in range(self.FGmax):
            if self.a0 + self.a1*i > Ex:
                break
        i2 = i
        i1 = i2-1
        num = 0
        temp = 0
        temp0 = 0
        tempL = 0
        tempH = 0
        if (rhotmopaw[i1] > 0) and (rhotmopaw[i2] > 0) and (rhotmopaw[i2] > rhotmopaw[i1]):
            temp0 = self.a1/(np.log(rhotmopaw[i2]) - np.log(rhotmopaw[i1]))
            num = num + 1
        if (i1 > 0) and (rhotmopaw[i1 - 1] > 0) and (rhotmopaw[i2 - 1] > 0) and (rhotmopaw[i2 - 1] > rhotmopaw[i1 - 1]):
            tempL = self.a1/(np.log(rhotmopaw[i2-1]) - np.log(rhotmopaw[i1-1]))
            num = num + 1
        if (i2 < 8192-1) and (rhotmopaw[i1 + 1] > 0) and (rhotmopaw[i2 + 1] > 0) and (rhotmopaw[i2 + 1] > rhotmopaw[i1 + 1]):
            tempH = self.a1/(np.log(rhotmopaw[i2+1]) - np.log(rhotmopaw[i1+1]))
            num = num + 1
        if num > 0:
            temp = (tempL + temp0 + tempH)/num
        return temp
    
    def spin_distribution(self, Ex, I, sigma2):
        for i in range(self.FGmax):
            if self.a0 + self.a1*i > Ex:
                break
        i2 = i
        i1 = i2-1
        s2 = sigma2[i1] + ((sigma2[i2] - sigma2[i1])/self.a1) * (Ex - (self.a0 + self.a1*i1) )
        s2 = max(s2, 1.0)
        g_Ex = (2*I + 1)*np.exp(-(I+0.5)**2/(2*s2)) / (2*s2)
        if g_Ex > 1e-20:
            return g_Ex
        else:
            return 0.
    
    def corrL(self, L1, L2, alpha, Anorm):
        corrbest = 0
        corr = 0.25
        sumbest = 1e21
        free = L2 - L1
        if free <= 0:
            free = 1
        for j in range(3751):
            corr = corr + 0.001
            sum = 0.0
            for i in range(L1, L2+1):
                Ex = self.a0 + self.a1*i
                rhoL = max(self.nld_lvl[i, 1], 0.01)
                cc = corr * Anorm * np.exp(alpha*Ex) * self.rhopaw[i, 1]
                dc2 = (corr * Anorm * np.exp(alpha*Ex) * self.rhopaw[i, 2])**2
                dc2 = np.sqrt(dc2*dc2 + 1)
                if dc2 > 0:
                    sum += (cc - rhoL) * (cc - rhoL) / dc2
            
            sum = sum/free
            #if j == 499:
            #    sum0 = sum
            if sum <= sumbest:
                sumbest = sum
                corrbest= corr
            
        return corrbest
    
    def corrH(self, H1, H2, alpha, Anorm, a, T, E1, E0, extr_model, b1, b2, eta):
        corrbest = 0
        corr = 0.25
        sumbest = 1e21
        free = H2 - H1
        if free <= 0:
            free = 1
        for j in range(3751):
            corr = corr + 0.001
            sum = 0.0
            for i in range(H1, H2+1):
                Ex = self.a0 + self.a1*i
                cc = corr * Anorm * np.exp(alpha*Ex) * self.rhopaw[i, 1]
                dc2 = (corr * Anorm * np.exp(alpha*Ex) * self.rhopaw[i, 2])**2
                _, rhox = self.rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)
                dc2 = np.sqrt(dc2*dc2 + 1)
                if dc2 > 0:
                    sum += (cc - eta * rhox)**2 / dc2
            sum = sum/free
            #if j == 499:
            #    sum0 = sum
            if sum <= sumbest:
                sumbest = sum
                corrbest = corr
        
        return corrbest
    
    def extendL(self, TL1, TL2, sigdimBn, nsig):
        nsigL = np.zeros((8192,3))
        steps = 1000
        abest = 0
        bbest = 0
        x1 = self.a0 + self.a1*TL1
        x2 = self.a0 + self.a1*TL2
        y1 = np.log(nsig[TL1, 1])
        y2 = np.log(nsig[TL2, 1]) 
        if TL2 - TL1 > 3:
            x1 = self.a0 + self.a1*TL1 + 0.5
            x2 = self.a0 + self.a1*TL2 + 0.5
            y1 = (np.log(nsig[TL1, 1]) + np.log(nsig[TL1 + 1, 1]))/2
            y2 = (np.log(nsig[TL2, 1]) + np.log(nsig[TL2 - 1, 1]))/2
        ai = (y2-y1)/x2-x1
        bi = y1 - ai * x1
        al = ai/3
        ah = ai*3
        astep = (ah - al)/steps
        bh = bi + 2 * ai * (x2 - x1)
        bl = bi - 2 * ai * (x2 - x1)
        bstep = (bh - bl)/steps
        
        chibest = 999999999.9
        bb = bl
        for i in range(steps):
            bb = bb + bstep
            aa = al 
            for j in range(steps):
                aa = aa + astep
                chi = 0
                for k in range(TL1, TL2+1):
                    if (nsig[k, 1] <= 0) or (nsig[k, 2] <= 0):
                        break
                    x = self.a0 + self.a1*k
                    y = aa * x + bb
                    yi = np.log(nsig[k,1])
                    dyi = nsig[k,2]/nsig[k,1]
                    chi = chi + (y-yi)**2/dyi**2
                chi = chi/(TL2-TL1)
                if (chi < chibest) and (chi > 0):
                    chibest = chi
                    abest = aa
                    bbest = bb
        abestL = abest - 3/x1
        bbestL = bbest + 3*(1 - np.log(x1))
        for i in range(sigdimBn):
            x = self.a0 + self.a1*i
            nsigL[i,1] = np.exp(abest * x + bbest)
            if i < TL1:
                nsigL[i,1] = x**3 * np.exp(abestL * x + bbestL)
        return nsigL
    
    def extendH(self, TH1, TH2, FGmax, nsig):
        nsigH = np.zeros((8192,3))
        steps = 1000
        abest = 0
        bbest = 0
        x1 = self.a0 + self.a1*TH1
        x2 = self.a0 + self.a1*TH2
        y1 = np.log(nsig[TH1, 1])
        y2 = np.log(nsig[TH2, 1]) 
        if TH2 - TH1 > 3:
            x1 = self.a0 + self.a1*TH1 + 0.5
            x2 = self.a0 + self.a1*TH2 + 0.5
            y1 = (np.log(nsig[TH1, 1]) + np.log(nsig[TH1 + 1, 1]))/2
            y2 = (np.log(nsig[TH2, 1]) + np.log(nsig[TH2 - 1, 1]))/2
        ai = (y2-y1)/x2-x1
        bi = y1 - ai * x1
        al = ai/3
        ah = ai*3
        astep = (ah - al)/steps
        bh = bi + 2 * ai * (x2 - x1)
        bl = bi - 2 * ai * (x2 - x1)
        bstep = (bh - bl)/steps
        
        chibest = 999999999.9
        bb = bl
        for i in range(steps):
            bb = bb + bstep
            aa = al 
            for j in range(steps):
                aa = aa + astep
                chi = 0
                for k in range(TH1, TH2 + 1):
                    if (nsig[k, 1] <= 0) or (nsig[k, 2] <= 0):
                        break
                    x = self.a0 + self.a1*k
                    y = aa * x + bb
                    yi = np.log(nsig[k,1])
                    dyi = nsig[k,2]/nsig[k,1]
                    chi = chi + (y-yi)**2/dyi**2
                chi = chi/(TH2-TH1)
                if (chi < chibest) and (chi > 0):
                    chibest = chi
                    abest = aa
                    bbest = bb
        abestH = abest
        bbestH = bbest
        for i in range(int(FGmax/5)):
            x = self.a0 + self.a1*i
            nsigH[i,1] = np.exp(abestH * x + bbestH)
        return nsigH
    
    def rhofg(self, ex, a, T, E1, E0, imodel, b1, b2):
        rhox = 0.1
        uCT = max(0.005, ex - E0)
        uFG = max(0.005, ex - E1)
        sig2 = max(0.5, b1 + b2*ex)
        
        if imodel == 1:
            rhox = np.exp(uCT/T)/T
        if imodel == 2:
            uFG = max((25./16.)/a, uFG)
            rhox = np.exp(2*np.sqrt(a*uFG))/(12*np.sqrt(2*sig2)*a**0.25*uFG**1.25)
        return sig2, rhox
        
    def write_input_nrm(self, Gg, curr_nld): #TODO:obsolete?
        '''
        Writes the normalization.c input file
        '''
        lines = []
        lines.append(['0', '{:.6f}'.format(self.Sn), '{:.6f}'.format(self.target_spin)])
        lines.append(['{:.6f}'.format(curr_nld.D0), '{:.6f}'.format(Gg)])
        lines.append(['105.000000', '{:.6f}'.format(curr_nld.FWHM)])
        
        newinput_nrm = ''
        for line in lines:
            curr_line = ' '
            for el in line:
                curr_line += (el + ' ')
            curr_line += '\n'
            newinput_nrm += curr_line
        
        with open('input.nrm', 'w') as write_obj:
            write_obj.write(newinput_nrm)
            
    def normalization_integral(self, curr_nld):
        
        '''
        runs the normalization.c integral, now translated in Python,
        and returns the integral result
        '''
        Eres = 400.
        de = 10.
        Bn_keV = self.Sn*1000
        It = self.target_spin
        eg = 0
        
        if It == 0.0:
            while eg < Bn_keV + Eres:
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int3 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int4 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                eg += de
            Int = int3 + int4
            return Int
            
            """
            def integrand(eg, Bn_keV, It, curr_nld):
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int1 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int2 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                return int1 + int2
            
            Int = quad(integrand, 0.0, Bn_keV + Eres, args=(Bn_keV, It, curr_nld), limit = int((Bn_keV + Eres)/de), epsabs = 1.0)
            """
        elif It == 0.5:
            while eg < Bn_keV + Eres:
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int2 = Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
                int3 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int4 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                eg += de
            Int = int2 + 2*int3 + int4
            return Int
            
            """
            def integrand(eg, Bn_keV, It, curr_nld):
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int1 = Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
                int2 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int3 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                return int1 + 2*int2 + int3
            Int = quad(integrand, 0.0, Bn_keV + Eres, args=(Bn_keV, It, curr_nld), limit = int((Bn_keV + Eres)/de), epsabs = 1.0)
            """
        elif It == 1.0:
            while eg < Bn_keV + Eres:
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int2 = Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
                int3 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int4 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                eg += de
            Int = 2*int2 + 2*int3 + int4
            return Int
            
            """
            def integrand(eg, Bn_keV, It, curr_nld):
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int1 = Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
                int2 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int3 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                return 2*int1 + 2*int2 + int3
            Int = quad(integrand, 0.0, Bn_keV + Eres, args=(Bn_keV, It, curr_nld), limit = int((Bn_keV + Eres)/de), epsabs = 1.0)
            """
        elif It > 1.0:
            while eg < Bn_keV + Eres:
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int1 = Teg*rhoex * (It / sigex) * np.exp(-(It - 1)**2 / sigex)
                int2 = Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
                int3 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int4 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                eg += de
            Int = int1 + 2*int2 + 2*int3 + int4
            return Int
            
            
            
            """
            def integrand(eg, Bn_keV, It, curr_nld):
                ex = Bn_keV - eg
                Teg = self.T_eg(eg, curr_nld)
                rhoex = self.rho_ex(ex, curr_nld)
                sigex = self.sig_ex(ex, curr_nld)
                int1 = Teg*rhoex * (It / sigex) * np.exp(-(It - 1)**2 / sigex)
                int2 = Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
                int3 = Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
                int4 = Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
                return int1 + 2*int2 + 2*int3 + int4
            
            Int = quad(integrand, 0.0, Bn_keV + Eres, args=(Bn_keV, It, curr_nld), limit = int((Bn_keV + Eres)/de), epsabs = 1.0)
            
            return Int[0]
            """

    def run_GSF_sim(self, current_Gg, curr_nld):
        
        Int = curr_nld.Int
        
        D0 = curr_nld.D0
        #Int = Int * de
        Fac = Int * D0 / current_Gg
        
        #write gsf
        gsf_rawmat = np.zeros_like(curr_nld.sig_mat)
        for i in range(len(gsf_rawmat)):
            Eg = self.a0 + self.a1*i
            gsf_rawmat[i,0] = Eg
            if Eg > 0:
                if Fac == 0.0:
                    gsf_rawmat[i,1] = np.nan
                    gsf_rawmat[i,2] = np.nan
                else:
                    gsf_rawmat[i,1] = curr_nld.sig_mat[i,1]/(Fac*Eg**3)
                    gsf_rawmat[i,2] = curr_nld.sig_mat[i,2]/(Fac*Eg**3)
        
        curr_gsf = gsf_p(gsf_rawmat)
        curr_gsf.clean_nans()
        curr_gsf.Bnorm = Fac
        curr_gsf.D0 = curr_nld.D0
        curr_gsf.FWHM = curr_nld.FWHM
        curr_gsf.Gg = current_Gg
        curr_gsf.nld = curr_nld
        curr_gsf.chi2 = curr_nld.chi2 + ((self.Gg - curr_gsf.Gg)/self.dGg)**2
        
        return curr_gsf
        
    
    def T_eg(self, eg, curr_nld):
        i2 = -1
        for ii in range(self.NchBn):
            if eg > (self.a0 + self.a1*ii)*1000:
                i2 = ii + 1
        if i2 == -1 or i2 > self.NchBn-1:
            i2 = self.NchBn - 1
        if i2 == 0:
            i2 = 1
        i1 = i2 - 1
        eg1 = (self.a0 + self.a1*i1)*1000
        eg2 = (self.a0 + self.a1*i2)*1000
        Teg = curr_nld.sigpawext[i1] + (curr_nld.sigpawext[i2] - curr_nld.sigpawext[i1])*(eg-eg1)/(eg2-eg1)
        Teg = max(Teg, 1e-10)
        return Teg
    
    def rho_ex(self, ex, curr_nld):
        i2 = -1
        for ii in range(self.NchBn):
            if ex > (self.a0 + self.a1*ii)*1000:
                i2 = ii + 1
        if i2 == -1 or i2 > self.NchBn-1:
            i2 = self.NchBn - 1
        if i2 == 0:
            i2 = 1
        i1 = i2 - 1
        ex1 = (self.a0 + self.a1*i1)*1000
        ex2 = (self.a0 + self.a1*i2)*1000
        rhox = curr_nld.rhotmopaw[i1] + (curr_nld.rhotmopaw[i2] - curr_nld.rhotmopaw[i1])*(ex - ex1)/(ex2 - ex1)
        rhox = max(rhox, 0)
        return rhox
    
    def sig_ex(self, ex, curr_nld):
        i2 = -1
        for ii in range(self.NchBn):
            if ex > (self.a0 + self.a1*ii)*1000:
                i2 = ii + 1
        if i2 == -1 or i2 > self.NchBn-1:
            i2 = self.NchBn - 1
        if i2 == 0:
            i2 = 1
        i1 = i2 - 1
        ex1 = (self.a0 + self.a1*i1)*1000
        ex2 = (self.a0 + self.a1*i2)*1000
        sigx = 2*curr_nld.spincut[i1]**2 + (2*curr_nld.spincut[i2]**2 - 2*curr_nld.spincut[i1]**2)*(ex - ex1)/(ex2 - ex1)
        sigx = max(sigx, 0.01)
        return sigx
    
    def initialize_nld_inputlist(self): #TODO: obsolete?
        '''
        Load the input list for counting.c with default values
        '''
        inputlist = [self.D0, self.L1, self.L2, self.H1, self.H2, self.TL1, self.TL2, self.TH1, self.TH2, self.extr_model, self.Ex_low, self.s_low, self.sigma, self.FWHM, self.a, self.E1, self.T, self.E0]
        return inputlist
    
    def make_MC_func(self, nld_inputlists, Ggs):
        
        def MC_func(i):
            curr_gsfs = []
            nld_inputlist = nld_inputlists[i]
            curr_nld = self.run_NLD_sim(nld_inputlist)
            for Gg in Ggs:
                curr_gsfs.append(self.run_GSF_sim(Gg, curr_nld))
            return output_pair(curr_nld, curr_gsfs)
        return MC_func
    
    def prompt_grid_search(self, variable, lowlim, highlim, num, N_cores):
        '''
        runs the small grid search at the beginning
        '''
        
        nld_inputlist = self.initialize_nld_inputlist()
        
        nld_inputlists = []
        
        for s in np.linspace(lowlim, highlim, num = num):
            new_inputlist = nld_inputlist.copy()
            new_inputlist[NLD_keys.index(variable)] = s
            nld_inputlists.append(new_inputlist)
        
        MC_sim = self.make_MC_func(nld_inputlists, [self.Gg])
        
        p = Pool(N_cores)
        result = list(tqdm.tqdm(p.imap(MC_sim, range(num)), total=num, desc = 'Quick grid search'))
        res_matr = np.array(result)
        
        self.nlds += [el.nld for el in res_matr]
        self.gsfs += [el.gsf[0] for el in res_matr]
        
        
        '''
        for s in tqdm.tqdm(np.linspace(lowlim, highlim, num = num), desc = 'Quick grid search'):
        #for s in np.linspace(lowlim, highlim, num = num):
            nld_inputlist[NLD_keys.index(variable)] = s
            
            curr_nld = self.run_NLD_sim(nld_inputlist)
            curr_gsf = self.run_GSF_sim(self.Gg, curr_nld)
            self.nlds.append(curr_nld)
            self.gsfs.append(curr_gsf)
        '''
    
    def MC_normalization(self, N_cores = 4, opt_range = [0.9,1.1], MC_range = 1000, load_lists = False, num = 20, delete_points_nld = None, delete_points_gsf = None):
        
        '''
        Runs the MC simulations, by picking random values for the input variables,
        running the counting.c and normalization.c programs, storing the results
        in objects, assign chi2 scores to them and saving them into lists.
        If you already have run MC_normalization and you already have the lists, 
        set load_lists = True.
        opt_range gives the range  within which the mini grid search will be run,
        an num gives the number of simulations in the mini gridsearch.
        MC_range is the number of Monte Carlo simulations
        '''
        
        if load_lists:
            self.nlds = np.load('nlds.npy', allow_pickle = True)
            self.gsfs = np.load('gsfs.npy', allow_pickle = True)
        else:
            self.MC_range = MC_range
            
            #initialize lists, start calculating NLDs and GSFs
            self.nlds = []
            self.gsfs = []
            
            #first: a small grid search by varying only one of the rho variables at a time.
            self.grid_searches = 0
            self.num = num
            for variable in self.rho_variables:
                vardouble = getattr(self, variable + 'double')
                varflat = getattr(self, variable + 'flat')
                
                if vardouble and varflat:
                    lowlim = getattr(self, variable + 'min')*opt_range[0]
                    highlim = getattr(self, variable + 'max')*opt_range[1]
                    self.prompt_grid_search(variable, lowlim, highlim, num, N_cores)
                    self.grid_searches += 1
            
            nld_inputlists = []
            Ggs = []
            for i in range(MC_range):
                #initialize input list for NLD with default values
                nld_inputlist = self.initialize_nld_inputlist()
                current_Gg = self.Gg
                
                for couple in couples:
                    if (couple[0] in self.free_parameters) and (couple[1] in self.free_parameters) and (getattr(self, couple[0] + '_range') == getattr(self, couple[1] + '_range')):
                        par_range = getattr(self, couple[0] + '_range')
                        values = np.sort(rng.choice(np.arange(par_range[0], par_range[1] + 1), size = 2, replace = False))
                        nld_inputlist[NLD_keys.index(couple[0])] = values[0]
                        nld_inputlist[NLD_keys.index(couple[1])] = values[1]
                
                #extr_model can only be 1 or 2. If this is a free parameter, 1 or 2 will be picked at random.
                if 'extr_model' in self.free_parameters:
                    nld_inputlist[NLD_keys.index('extr_model')] = rng.integers(1,3)
                    
                #finally, the floats
                float_pars = ['D0', 'Ex_low', 's_low', 'sigma', 'FWHM', 'a', 'E1', 'T', 'E0']
                for float_par in float_pars:
                    if float_par in self.free_parameters:
                        par_range = getattr(self, float_par + '_range')
                        nld_inputlist[NLD_keys.index(float_par)] = rng.uniform(low = par_range[0], high = par_range[1])
                
                nld_inputlists.append(nld_inputlist)
                
                if 'Gg' in self.free_parameters:
                    par_range = getattr(self, 'Gg_range')
                    #for each NLD simulation, run 10 GSF simulations (it seems like I need it to get enough simulations, and the most effective way instead of incresing MC_range)
                    for j in range(10):
                        current_Gg = rng.uniform(low = par_range[0], high = par_range[1])
                        Ggs.append(current_Gg)
                else:
                    Ggs.append(self.Gg)
            
            MC_sim = self.make_MC_func(nld_inputlists, Ggs)
            p = Pool(N_cores)
            result = list(tqdm.tqdm(p.imap(MC_sim, range(MC_range)), total=MC_range, desc = 'Normalizing NLDs and GSFs'))
            res_matr = np.array(result)
            
            self.nlds += [el.nld for el in res_matr]
            self.gsfs += [el.gsf[i] for i in range(10) for el in res_matr]
            
            if isinstance(delete_points_nld, list):
                for i, el in enumerate(self.nlds):
                    el.delete_point(delete_points_nld)
            if isinstance(delete_points_gsf, list):
                for i, el in enumerate(self.gsfs):
                    el.delete_point(delete_points_gsf)
            
            np.save('nlds.npy', self.nlds)
            np.save('gsfs.npy', self.gsfs)
    
    
    
    def calc_TALYS_models(self, load_lists = False, N_cores = 4, number_of_strength_models = 9):
        '''
        calculate the TALYS NLD and GSF models for comparison with the normalized NLD and GSF
        '''
        self.TALYS_models = True
        if self.TALYS_ver:
            assert self.TALYS_ver in supported_TALYS_versions, f'TALYS {self.TALYS_ver} version not supported'
        else:
            raise Exception('remember to define the TALYS version with "set_TALYS_version"')
        
        if load_lists:
            self.nld_models = np.load('nld_models.npy', allow_pickle = True)
            self.gsf_models = np.load('gsf_models.npy', allow_pickle = True)
        else:
            os.makedirs('talys_models', exist_ok = True) 
            os.chdir('talys_models')
            
            talys_sim = make_talys_sim_simple(self.A, self.Z, talys_executable_path = self.talys_executable_path, TALYS_ver = self.TALYS_ver) #make function
            p = Pool(N_cores)
            length = number_of_strength_models
            result = list(tqdm.tqdm(p.imap(talys_sim, range(length)), total=length, desc = 'Calculating TALYS models'))
            os.chdir(root_folder)
            output_list = np.array(result)
            self.nld_models = [n.nld for n in output_list[:6]]
            self.gsf_models = [n.gsf for n in output_list]
            
            #save models to file
            np.save('nld_models.npy', self.nld_models)
            np.save('gsf_models.npy', self.gsf_models)
            
            #delete temp files
            #shutil.rmtree('talys_models')
    
    def write_NLD_GSF_tables(self, path = ''):
        '''
        find the errors from the simulations, and write human readable tables
        '''
        best_fits = []
        valmatrices = []
        for i, (lst, lab) in enumerate(zip([self.nlds, self.gsfs], ['nld','gsf'])):
            valmatrix, best_fit = calc_errors_chis(lst, lab = lab, return_best_fit=True)
            valmatrix = clean_valmatrix(valmatrix)
            valmatrices.append(valmatrix)
            best_fits.append(best_fit)
            header = 'Energy [MeV], best_fit, best_fit-sigma, best_fit+sigma, staterr' 
            if (path != ''):
                if path[-1] != '/':
                    path = path + '/'
            np.savetxt(path + lab + '_whole.txt', valmatrix, header = header)
        self.best_fitting_nld = best_fits[0]
        self.best_fitting_gsf = best_fits[1]
        self.NLD_table = valmatrices[0]
        self.GSF_table = valmatrices[1]

        np.save(path + 'best_fits.npy', best_fits)
        
    def plot_graphs(self, save = True):
        
        scaling_factor = 1.5#0.9
        fig0, ax0 = plt.subplots(figsize = (5.0*scaling_factor, 3.75*scaling_factor), dpi = 300)
        fig1, ax1 = plt.subplots(figsize = (5.0*scaling_factor, 3.75*scaling_factor), dpi = 300)
        
        Sn_lowerr = self.NLD_table[-1,1] - self.NLD_table[-1,2]
        Sn_higherr = self.NLD_table[-1,3] - self.NLD_table[-1,1]
        ax0.errorbar(self.Sn, self.NLD_table[-1,1], yerr=np.array([[Sn_lowerr, Sn_higherr]]).T, ecolor='g',linestyle=None, elinewidth = 4, capsize = 5, label=r'$\rho$ at Sn')
        
        ax0.fill_between(self.NLD_table[:-1,0], self.NLD_table[:-1,3], self.NLD_table[:-1,-3], color = 'b', alpha = 0.2, label=r'1$\sigma$ conf.')
        ax0.plot(self.nld_lvl[:,0], self.nld_lvl[:,1], 'k-', label='Known lvs.')
        ax0.axvspan(self.ExL1, self.ExL2, alpha=0.2, color='red',label='Fitting intv.')
        ax0.errorbar(self.NLD_table[:-1,0], self.NLD_table[:-1,1],yerr=self.NLD_table[:-1,-1], fmt = '.', color = 'b', ecolor='b', label='This work')
        ax0.plot(self.best_fitting_nld.extr_mat[(self.H1-5):,0], self.best_fitting_nld.extr_mat[(self.H1-5):,1], color = 'k', linestyle = '--', alpha = 1, label = 'extrap.')
        
        #Debug
        #extr2 = [self.rhofg(ex, self.best_fitting_nld.a, self.best_fitting_nld.T, self.best_fitting_nld.E1, self.best_fitting_nld.E0, self.best_fitting_nld.extr_model, self.best_fitting_nld.b1, self.best_fitting_nld.b2) for ex in self.best_fitting_nld.extr_mat[:,0]]
        #ax0.plot(self.best_fitting_nld.extr_mat[(self.H1-5):,0], extr2[(self.H1-5):], color = 'b', linestyle = '--', alpha = 1, label = 'extrap.2')
        
        ax1.fill_between(self.GSF_table[:,0], self.GSF_table[:,3], self.GSF_table[:,-3], color = 'b', alpha = 0.2, label=r'1$\sigma$ conf.')
        if self.TALYS_models:
            cmap = matplotlib.cm.get_cmap('YlGnBu')
            stls = ['-','--','-.',':','-','--','-.',':','-']
            
            #plot TALYS gsf
            for i, TALYS_strength in enumerate(self.gsf_models):
                if i < 3:
                    col = 3
                elif i < 6:
                    col = 6
                else:
                    col = 8
                ax1.plot(TALYS_strength[:,0],TALYS_strength[:,1] + TALYS_strength[:,2], color = cmap(col/9), linestyle = stls[i], alpha = 0.8, label = 'strength %d'%(i+1))
            
            #plot TALYS nld
            for i, TALYS_ldmodel in enumerate(self.nld_models):
                if i<3:
                    col = 3
                else:
                    col = 5
                ax0.plot(TALYS_ldmodel[:,0],TALYS_ldmodel[:,3], color = cmap(col/6), linestyle = stls[i], alpha = 0.8, label = 'ldmodel %d'%(i+1))
        
        ax1.errorbar(self.GSF_table[:,0], self.GSF_table[:,1],yerr=self.GSF_table[:,-1], fmt = '.', color = 'b', ecolor='b', label='This work')
        
        #Plot experiment data
        for ax in [ax0,ax1]:
            ax.set_yscale('log')
            ax.legend(frameon = False)
        ax0.set_xlabel(r'$E_x$ [MeV]')
        ax1.set_xlabel(r'$E_\gamma$ [MeV]')
        ax0.set_ylabel(r'NLD [MeV$^{-1}$]')
        ax1.set_ylabel(r'GSF [MeV$^{-3}$]')
        ax0.set_ylim([min(self.NLD_table[:-1,-3])*0.5, max(self.NLD_table[:,3])*2])
        ax0.set_xlim([self.NLD_table[0,0]-0.5, self.NLD_table[-1,0]+0.5])
        ax1.set_ylim([min(self.GSF_table[:,-3])*0.5, max(self.GSF_table[:,3])*2])
        ax1.set_xlim([self.GSF_table[0,0]-0.5, self.GSF_table[-1,0]+0.5])
        fig0.tight_layout()
        fig1.tight_layout()
        fig0.show()
        fig1.show()
        if save:
            fig0.savefig('nld.pdf', format = 'pdf')
            fig1.savefig('gsf.pdf', format = 'pdf')
            fig0.savefig('nld.png', format = 'png')
            fig1.savefig('gsf.png', format = 'png')
            
    def write_ncrate_MACS_tables(self, path = '', load_lists = False, label = ''):
        '''
        find the errors from the simulations, and write human readable tables
        '''
        
        if label != '':
            label = '_' + label
        best_fits = []
        valmatrices = []
        if load_lists:
            self.ncrates = np.load(f'ncrates{label}.npy', allow_pickle = True)
            self.MACSs = np.load(f'MACSs{label}.npy', allow_pickle = True)
        
        for i, (lst, lab) in enumerate(zip([self.ncrates, self.MACSs], ['ncrate','MACS'])):
            valmatrix, best_fit = calc_errors_chis(lst, lab = lab, return_best_fit=True)
            valmatrix = np.delete(valmatrix, -1, 1)
            valmatrices.append(valmatrix)
            best_fits.append(best_fit)
            if lab == 'ncrate':
                valmatrix = np.append(valmatrix, self.ncrate_yerr, 1)
                header = 'Temperature [GK], n-capture rate [cm3/mol/s], best_fit-sigma, best_fit+sigma, best_fit - low staterr, best_fit + high staterr' 
            elif lab == 'MACS':
                valmatrix = np.append(valmatrix, self.MACS_yerr, 1)
                header = 'T*k_B [keV], MACS [mb], best_fit-sigma, best_fit+sigma, best_fit - low staterr, best_fit + high staterr' 
            if (path != ''):
                if path[-1] != '/':
                    path = path + '/'
            np.savetxt(path + lab + label + '_whole.txt', valmatrix, header = header)
        self.best_fitting_ncrate = best_fits[0]
        self.best_fitting_MACS = best_fits[1]
        self.ncrate_table = valmatrices[0]
        self.MACS_table = valmatrices[1]

        np.save(path + 'best_fits_ncrates.npy', best_fits)
        
    def run_TALYS_sims_parallel(self, M1, high_energy_interp, N_cores = 4, chi2_window = 1.0, load_lists = False, run_stat = True, jlmomp = False, label = ''):
        '''
        Run the TALYS simulations in parallel. This might take some hours, depending on how many simulations and how many cores you are using.
        Anything between ~3h and ~48h. Play around and see if you want to run it overnight or over a weekend, or on a different computer altogether.
        Input description, from the example files:
        "M1": this could be a float between 0 and 1, a list 3 elements long, or a list 6 elements long.
            if it's a float, the M1 strength will be simply calculated as a fraction of the total strength: i.e. if M1 = 0.1, then for every bin the strength will be divided between 10% M1 and 90% E1
            if it's a list of 3, these will be the centroid, the Gamma and the sigma of an SLO (e.g. of the spin-flip resonance)
            if it's a list of 6, these will be the centroid, the Gamma and the sigma of two SLOs (e.g., the spin-flip and the scissors resonances)
        "high_energy_extrap": a (2,n) array with energies and gsf for higher energies. For example, this could be the experimental GDR data for a nearby nucleus, or an exponential extrapolation
        "chi2_window" = a bit difficult to explain, but the recommended value is 1.0, or as high as possible, the maximum is 2.0. 
            The higher it is, the more simulations are run, and the longer time it takes, but the result is more precise. 
            Run once at e.g. chi2_window = 0.2 to get a rough result if you want something quick (although it could be a bit unstable, in that case, increase the number)
        '''
        if self.TALYS_ver:
            assert self.TALYS_ver in supported_TALYS_versions, 'TALYS version not supported'
        else:
            raise Exception('remember to define the TALYS version with "set_TALYS_version"')
        if label != '':
            label = '_' + label
        if load_lists:
            self.ncrates = np.load(f'ncrates{label}.npy', allow_pickle = True)
            self.MACSs = np.load(f'MACSs{label}.npy', allow_pickle = True)
            self.ncrate_yerr = np.load(f'ncrate_yerr{label}.npy', allow_pickle = True)
            self.MACS_yerr = np.load(f'MACS_yerr{label}.npy', allow_pickle = True)
        else:
            chimin = self.best_fitting_gsf.chi2
            filtered_gsfs = [el for el in self.gsfs if ((el.chi2 > chimin + 1 - chi2_window/2) and (el.chi2 < chimin + 1 + chi2_window/2))]
            filtered_gsfs.append(self.best_fitting_gsf)
            self.ncrates = []
            self.MACSs = []
            os.makedirs('talys_tmp', exist_ok = True) 
            os.chdir('talys_tmp')
            tab_filename = Z2Name(self.Z) + '.tab'
            print(f'Number of gsfs: {len(filtered_gsfs)}')
            params = {'filtered_gsfs': filtered_gsfs, 'tab_filename': tab_filename, 'A': self.A, 'Z': self.Z, 'M1': M1, 'high_energy_interp': high_energy_interp, 'jlmomp': jlmomp}
            talys_sim = make_talys_sim(params, talys_root_path=self.talys_root_path, talys_executable_path = self.talys_executable_path, TALYS_ver = self.TALYS_ver) #make function
            
            p = Pool(N_cores)
            length = len(filtered_gsfs)
            result = list(tqdm.tqdm(p.imap(talys_sim, range(length)), total=length, desc = 'Calculating systematic errors'))
            res_matr = np.array(result)
            
            self.ncrates = res_matr[:,0]
            self.MACSs = res_matr[:,1]
            os.chdir(root_folder)
            np.save(f'ncrates{label}.npy', self.ncrates)
            np.save(f'MACSs{label}.npy', self.MACSs)
        
            #delete temp files
            shutil.rmtree('talys_tmp')
            if run_stat:
                self.run_TALYS_statistical_errors(M1, high_energy_interp, jlmomp, N_cores = N_cores, label = label)
                
        
    def run_TALYS_statistical_errors(self, M1, high_energy_interp, jlmomp, label = '', N_cores = 4):
        '''
        Propagate the statistical errors from the experiment to the MACS and the ncrate by running four
        TALYS simulations.
        '''
        if self.TALYS_ver:
            assert self.TALYS_ver in supported_TALYS_versions, 'TALYS version not supported'
        else:
            raise Exception('remember to define the TALYS version with "set_TALYS_version"')
        if label != '':
            label = '_' + label
        stat_err_gsf_list = [copy.deepcopy(self.best_fitting_gsf) for i in range(4)]
        
        tab_down, tab_up = make_scaled_talys_nld_cnt(self.best_fitting_gsf.nld)
        for i, el in enumerate(stat_err_gsf_list):
            if i in [0,1]:
                el.y = el.y + el.yerr
            else:
                el.y = el.y - el.yerr
            if i in [0,2]:
                el.nld.y = el.nld.y + el.nld.yerr
                el.nld.talys_nld_cnt = tab_up
            else:
                el.nld.y = el.nld.y - el.nld.yerr
                el.nld.talys_nld_cnt = tab_down
        os.makedirs('talys_stat', exist_ok = True) 
        os.chdir('talys_stat')
        params = {'filtered_gsfs': stat_err_gsf_list, 'tab_filename': Z2Name(self.Z) + '.tab', 'A': self.A, 'Z': self.Z, 'M1': M1, 'high_energy_interp': high_energy_interp, 'jlmomp': jlmomp}
        talys_sim = make_talys_sim(params, talys_root_path = self.talys_root_path, talys_executable_path = self.talys_executable_path, TALYS_ver = self.TALYS_ver) #make function
        
        p = Pool(N_cores)
        length = len(stat_err_gsf_list)
        result = list(tqdm.tqdm(p.imap(talys_sim, range(length)), total=length, desc = 'Calculating statistical errors'))
        
        res_matr = np.array(result)
        ncrates_stat = res_matr[:,0]
        MACSs_stat = res_matr[:,1]
        os.chdir(root_folder)
        shutil.rmtree('talys_stat')
        
        ncrate_lowerr = np.zeros_like(ncrates_stat[0].y)
        ncrate_higherr = np.zeros_like(ncrates_stat[0].y)
        MACS_lowerr = np.zeros_like(MACSs_stat[0].y)
        MACS_higherr = np.zeros_like(MACSs_stat[0].y)
        
        for i in range(len(ncrates_stat[0].y)):
            ncrate_lowerr[i] = min([el.y[i] for el in ncrates_stat])
            ncrate_higherr[i] = max([el.y[i] for el in ncrates_stat])
            MACS_lowerr[i] = min([el.y[i] for el in MACSs_stat])
            MACS_higherr[i] = max([el.y[i] for el in MACSs_stat])
        self.ncrate_yerr = np.c_[ncrate_lowerr, ncrate_higherr]
        self.MACS_yerr = np.c_[MACS_lowerr, MACS_higherr]
        np.save(f'ncrate_yerr{label}.npy', self.ncrate_yerr)
        np.save(f'MACS_yerr{label}.npy', self.MACS_yerr)
        
#Useful classes and functions preparing for the parallel calculations

class output_pair:
    def __init__(self, ld, sf):
        self.nld = ld
        self.gsf = sf

def make_talys_sim(pars, talys_root_path, talys_executable_path, TALYS_ver):
    filtered_gsfs = pars['filtered_gsfs']
    tab_filename = pars['tab_filename']
    A = pars['A']
    Z = pars['Z']
    M1 = pars['M1']
    high_energy_interp = pars['high_energy_interp']
    jlmomp = pars['jlmomp']

    def talys_sim(i):
        #uncomment for debugging
        #start = time.time()
        el = filtered_gsfs[i]
        subdir_path = str(i)
        os.makedirs(subdir_path, exist_ok = True)
        os.chdir(subdir_path)
        shutil.copyfile(talys_root_path + '/structure/density/ground/goriely/' + tab_filename, tab_filename.lower())
        make_TALYS_tab_file(tab_filename.lower(), el.nld.talys_nld_cnt, A, Z)
        make_E1_M1_files_simple(el.x, el.y, A, Z, M1 = M1, target_folder = '.', high_energy_interp=high_energy_interp, delete_points = None, units = 'mb')
        write_TALYS_inputfile(A, Z, TALYS_ver, jlmomp, target_dir = '.')
        shutil.copyfile(talys_executable_path, 'talys')
        os.system('chmod +x talys')
        os.system('./talys <talys.inp> talys.out')
        curr_ncrate = ncrate('astrorate.g')
        curr_MACS = MACS('astrorate.g')
        curr_ncrate.chi2 = curr_MACS.chi2 = el.chi2
        os.chdir('..')
        
        #finish = time.time()
        #print(f'simulation {i} finished in {finish - start} seconds')
        return [curr_ncrate, curr_MACS]
    return talys_sim

def make_talys_sim_simple(A, Z, talys_executable_path, TALYS_ver):
    
    def talys_sim(i):
        
        ldmodel = i + 1
        strength = i + 1
        if ldmodel > 6:
            ldmodel = 1
        subdir_path = str(i)
        os.makedirs(subdir_path, exist_ok = True)
        os.chdir(subdir_path)
        target_dir = '.'
        gnormstring = '1.'
        if TALYS_ver in ['1.96', '2.00']:
            if TALYS_ver == '2.00':
                gnormstring = 'n'
            inputfile = f'projectile n\nelement {Z2Name(Z)}\nmass {str(A - 1)}\nenergy n0-20.grid\nldmodel {ldmodel}\nwtable {Z} {A} 1.0 E1\nstrength {strength}\nlocalomp y\noutgamma y\nfilepsf y\noutdensity y\ngnorm {gnormstring}'
        else:
            print('TALYS version not supported')
        
        with open(f'{target_dir}/talys.inp', 'w') as write_obj:
            write_obj.write(inputfile)
        shutil.copyfile(talys_executable_path, 'talys')
        os.system('chmod +x talys')
        os.system('./talys <talys.inp> talys.out')
        output_gsf = readstrength(A, Z)
        output_nld = readldmodel_path('talys.out',A, Z)
        os.chdir('..')
        curr_output_pair = output_pair(output_nld, output_gsf)
        return curr_output_pair
    return talys_sim
    
def write_TALYS_inputfile(A, Z, TALYS_ver, jlmomp, target_dir = '.'):
    tab_filename = Z2Name(Z) + '.tab'
    omp_line = 'localomp y'
    if jlmomp:
        omp_line = 'jlmomp y'
    if TALYS_ver == '1.96':
        inputfile = f'projectile n\nelement {Z2Name(Z)}\nmass {str(A - 1)}\nenergy 1\nstrength 4\nstrengthm1 2\nE1file {Z} {A} {target_dir}/gsfE1.dat\nM1file {Z} {A} {target_dir}/gsfM1.dat\ngnorm 1.\ndensfile {Z} {A} {target_dir}/{tab_filename.lower()}\nptable {Z} {A} 0.0\nctable {Z} {A} 0.0\nNlevels {Z} {A} 30\nwtable {Z} {A} 1.0 E1\n{omp_line}\nupbend n\noutgamma y\nastro y'
    elif TALYS_ver == '2.00':   
        inputfile = f'projectile n\nelement {Z2Name(Z)}\nmass {str(A - 1)}\nenergy 1\ngnorm n\nE1file {Z} {A} {target_dir}/gsfE1.dat\nM1file {Z} {A} {target_dir}/gsfM1.dat\ndensfile {Z} {A} {target_dir}/{tab_filename.lower()}\nptable {Z} {A} 0.0\nctable {Z} {A} 0.0\nNlevels {Z} {A} 30\nwtable {Z} {A} 1.0 E1\n{omp_line}\nupbend n\noutgamma y\nastro y'
    else:
        print('TALYS version not supported')
    
    with open(f'{target_dir}/talys.inp', 'w') as write_obj:
        write_obj.write(inputfile)