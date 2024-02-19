#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:34:18 2024

@author: francesco
"""

import numpy as np
from systlib import D2rho, drho, import_ocl, chisquared, nld, gsf, import_Anorm_alpha, import_Bnorm, calc_errors_chis, clean_valmatrix
import matplotlib.pyplot as plt
from subprocess import call
import os
rng = np.random.default_rng(seed=1070)
#full adress of modified counting and normalization codes
counting_code_path = "/home/francesco/oslo-method-software-auto/prog/counting"
normalization_code_path = "/home/francesco/oslo-method-software-auto/prog/normalization"
rsc_folder = 'rhosigchi' #folder where you have run until rhosigchi and calculations will be run in
root_folder = os.getcwd()


NLD_keys = ['D0', 'L1', 'L2', 'H1', 'H2', 'TL1', 'TL2', 'TH1', 'TH2', 'extr_model', 'Ex_low', 's_low', 'sigma', 'FWHM']

couples = [['L1', 'L2'],
           ['H1', 'H2'],
           ['TL1', 'TL2'],
           ['TH1', 'TH2']]

class normalization:
    def __init__(self, OM_folderpath, a0, a1, A, Sn):
        self.a0 = a0
        self.a1 = a1
        self.OM_folderpath = OM_folderpath
        self.A = A
        self.Sn = Sn
        
    def set_attributes(self, **kwargs):
        '''
        attributes that are useful are
        sigma: spin-cutoff parameter. It can be one value, or a tuple/list in case one wants to take in consideration both FG and RMI
        dsigma: uncertainty in the spin-cutoff parameter. Float or tuple/list
        flat (boolean): if True, the probability of the spin-cutoff parameter to be between the two sigmas provided is the same as for getting one of the two sigmas. (if a normal error distribution is a Gaussian, with flat = True the distribution is Gaussian outside the interval between the sigmas, and flat within.)
        D0: average level spacing
        dD0: uncertainty in the average level spacing
        target_spin: spin in the N-1 nucleus
        Gg: average radiative strength (meV)
        dGg: uncertainty in average radiative strength (meV)
        
        NLD FITTING REGIONS
         - choose whether to give the lower and upper limit for the lower excitation energy fitting interval in energies (Ex1, Ex2), or bin numbers (L1, L2)
        Ex1: lower limit for the low excitation energy interval where the NLD is fitted (energy (MeV), float)
        Ex2: upper limit for the low excitation energy interval where the NLD is fitted (energy (MeV), float)
        or 
        L1: lower limit for the low excitation energy interval where the NLD is fitted (bin, int)
        L2: upper limit for the low excitation energy interval where the NLD is fitted (bin, int)
        
         - choose whether to give the lower and upper limit for the higher excitation energy fitting interval in energies (Ex3, Ex4), or bin numbers (H1, H2)
        Ex3: lower limit for the high excitation energy interval where the NLD model is fitted (energy (MeV), float)
        Ex4: upper limit for the high excitation energy interval where the NLD model is fitted (energy (MeV), float)
        or 
        H1: lower limit for the high excitation energy interval where the NLD model is fitted (bin, int)
        H2: upper limit for the high excitation energy interval where the NLD model is fitted (bin, int)
        
        TRANSMISSION COEFFICIENT FITTING REGIONS
        - choose whether to give the lower and upper limit for the lower gamma energy fitting interval for the transmission coefficient in energies (Eg1, Eg2), or bin numbers (TL1, TL2)
        Eg1: lower limit for the low gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        Eg2: upper limit for the low gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        or 
        TL1: lower limit for the low gamma energy interval where the NLD is fitted (bin, int)
        TL2: upper limit for the low gamma energy interval where the NLD is fitted (bin, int)
       
        - choose whether to give the lower and upper limit for the higher gamma energy fitting interval for the transmission coefficient in energies (Eg3, Eg4), or bin numbers (TH1, TH2)
        Eg3: lower limit for the high gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        Eg4: upper limit for the high gamma energy interval where the transmission coefficient is fitted (energy (MeV), float)
        or 
        TH1: lower limit for the high gamma energy interval where the transmission coefficient is fitted (bin, int)
        TH2: upper limit for the high gamma energy interval where the transmission coefficient is fitted (bin, int)
        
        ds_low, ds_high: before running a MC simulation, a run with mean D0, Gg and low energy interval matching L1 and L2 is done, by only varying the
        spin-cutoff parameter within the two limit values. ds_low and ds_high define by how much we should move away from the two values when doing this
        "mini grid search"
        
        L1_range_low, l2_range_high: the range where the L1 and L2 will be randomly picked from. If not specified, standard values of L1-1 and L2+1 will be picked.
        
        extr_model: NLD model to extrapolate the data to Sn. Either 1 (constant temperature) or 2 (FG)
        
        
        TODO: not urgent, but: if f.ex. L1, L2, H1, H2, TL1... are not given, use the same algorithm as in counting to guess some values.
        '''
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        assert type(self.sigma) == type(self.dsigma)
        
        self.double = False
        if isinstance(self.sigma, list):
            self.double = True
        
        #if sigma is tuple/list, then categorize into min and max
        if self.double:
            if self.sigma[0] > self.sigma[1]:
                self.sigmamax = self.sigma[0]
                self.sigmamin = self.sigma[1]
                self.dsigmamax = self.dsigma[0]
                self.dsigmamin = self.dsigma[1]
            else:
                self.sigmamax = self.sigma[1]
                self.sigmamin = self.sigma[0]
                self.dsigmamax = self.dsigma[1]
                self.dsigmamin = self.dsigma[0]
            self.sigma = (self.sigmamax + self.sigmamin)/2
            self.dsigma = (np.sqrt(self.dsigmamax**2 + self.dsigmamin**2))/2
        
        #If L1, L2... given as energies, translate them into bins, and print something
        #I could not find a smarter way to set this up. Maybe by using variable_dictionary, defined above?
        if 'L1' not in kwargs:
            self.L1 = round((self.Ex1 - self.a0)/self.a1)
            self.ExL1 = self.a0 + self.a1*self.L1
            print(f'L1 energy {self.Ex1} rounded to {self.ExL1} to coincide with bin number {self.L1}')
        if 'L2' not in kwargs:
            self.L2 = round((self.Ex2 - self.a0)/self.a1)
            self.ExL2 = self.a0 + self.a1*self.L2
            print(f'L2 energy {self.Ex2} rounded to {self.ExL2} to coincide with bin number {self.L2}')
        
        if 'H1' not in kwargs:
            self.H1 = round((self.Ex3 - self.a0)/self.a1)
            self.ExH1 = self.a0 + self.a1*self.H1
            print(f'H1 energy {self.Ex3} rounded to {self.ExH1} to coincide with bin number {self.H1}')
        if 'H2' not in kwargs:
            self.H2 = round((self.Ex4 - self.a0)/self.a1)
            self.ExH2 = self.a0 + self.a1*self.H2
            print(f'H2 energy {self.Ex4} rounded to {self.ExH2} to coincide with bin number {self.H2}')
        
        if 'TL1' not in kwargs:
            self.TL1 = round((self.Eg1 - self.a0)/self.a1)
            self.EgTL1 = self.a0 + self.a1*self.TL1
            print(f'TL1 energy {self.Eg1} rounded to {self.EgTL1} to coincide with bin number {self.TL1}')
        if 'TL2' not in kwargs:
            self.TL2 = round((self.Eg2 - self.a0)/self.a1)
            self.EgTL2 = self.a0 + self.a1*self.TL2
            print(f'TL2 energy {self.Eg2} rounded to {self.EgTL2} to coincide with bin number {self.TL2}')
        
        if 'TH1' not in kwargs:
            self.TH1 = round((self.Eg3 - self.a0)/self.a1)
            self.EgTH1 = self.a0 + self.a1*self.TH1
            print(f'TH1 energy {self.Eg3} rounded to {self.EgTH1} to coincide with bin number {self.TH1}')
        if 'TH2' not in kwargs:
            self.TH2 = round((self.Eg4 - self.a0)/self.a1)
            self.EgTH2 = self.a0 + self.a1*self.TH2
            print(f'TH2 energy {self.Eg4} rounded to {self.EgTH2} to coincide with bin number {self.TH2}')
        
            
    def set_variation_intervals(self, std_devs, **kwargs):
        
        '''
        std_devs: how many standard deviations the parameter will be varied in the MC simulation.
        kwargs in the form
        'D0': [min, max],
        'L1': [min, max]...
        
        for extr_model, only 1 and 2 are possible anyway. If you want to pick randomly between them, put
        'extr_model: 'yes'
        
        
        numerical values for min, max
        if min or max = 'err', the error value provided in set_attributes will be used
        
        if you want L1 and L2 to take all possible values within a range, give them the same limits (same for H1,H2,TL1...)
        '''
        self.std_devs = std_devs
        self.free_parameters = []
        self.n_free_parameters = 0
        for key, value in kwargs.items():
            self.free_parameters.append(key)
            if key == 'flat':
                setattr(self, key, value)
            elif (key == 'sigma') and self.double:
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
        

    #find uncertainty distribution of rho
    def make_rhouncdist(self):
        # returns a gaussian or flat-gaussian function expressing the uncertainty in rho
        if self.double and not self.flat:
            spin_cutoff = self.sigma
            d_spin_cutoff = self.sigmamax - self.sigma + self.dsigmamax
            self.rhomean = D2rho(self.D0, self.target_spin, self.sigma)
            self.rhostddev = drho(self.target_spin, self.sigma, d_spin_cutoff, self.D0, self.dD0, rho = self.rhomean)
            self.rho_lowererr = self.rhomean - self.rhostddev
            self.rho_uppererr = self.rhomean + self.rhostddev
            def uncdistr(self, x):
                return np.exp(-np.power((x - self.rhomean)/self.rhostddev, 2.)/2)
            setattr(self, 'uncdistr', uncdistr.__get__(self, self.__class__))
        elif self.double and self.flat:
            rhos = np.zeros(2)
            drhos = np.zeros(2)
            for i, (spin_cutoff, dspin_cutoff) in enumerate(zip([self.sigmamin, self.sigmamax], [self.dsigmamin, self.dsigmamax])):
                rhos[i] = D2rho(self.D0, self.target_spin, spin_cutoff)
                drhos[i] = drho(self.target_spin, spin_cutoff, dspin_cutoff, self.D0, self.dD0)
            self.sortedrhos = np.sort(rhos)
            self.sorteddrhos = drhos
            if not np.array_equal(self.sortedrhos, rhos):
                self.sorteddrhos = np.flip(drhos)
            def uncdistr(self, xx):
                if isinstance(xx, float):
                    if xx < self.sortedrhos[0]:
                        return np.exp(-np.power((xx - self.sortedrhos[0])/self.sorteddrhos[0], 2.)/2)
                    elif xx < self.sortedrhos[1]:
                        return 1.0
                    else:
                        return np.exp(-np.power((xx - self.sortedrhos[1])/self.sorteddrhos[1], 2.)/2)
                else:
                    results = np.zeros_like(xx)
                    for i,x in enumerate(xx):
                        if x < self.sortedrhos[0]:
                            results[i] = np.exp(-np.power((x - self.sortedrhos[0])/self.sorteddrhos[0], 2.)/2)
                        elif x < self.sortedrhos[1]:
                            results[i] = 1.0
                        else:
                            results[i] = np.exp(-np.power((x - self.sortedrhos[1])/self.sorteddrhos[1], 2.)/2)
                    return results
            
            self.rhomean = (self.sortedrhos[0] - self.sorteddrhos[0] + self.sortedrhos[1] + self.sorteddrhos[1])/2
            self.rhostddev = self.sortedrhos[1] + self.sorteddrhos[1] - self.rhomean
            self.rho_lowererr = self.rhomean - self.rhostddev
            self.rho_uppererr = self.rhomean + self.rhostddev
            setattr(self, 'uncdistr', uncdistr.__get__(self, self.__class__))
        else:
            self.rhomean = D2rho(self.D0, self.target_spin, self.sigma)
            self.rhostddev = drho(self.target_spin, self.sigma, self.dsigma, self.D0, self.dD0)
            def uncdistr(self, x):
                return np.exp(-np.power((x - self.rhomean)/self.rhostddev, 2.)/2)
            self.rho_lowererr = self.rhomean - self.rhostddev
            self.rho_uppererr = self.rhomean + self.rhostddev
            setattr(self, 'uncdistr', uncdistr.__get__(self, self.__class__))
    
    def make_rho_chi2score(self):
        #translates the gaussian or flat-gaussian uncertainty distribution into a chi2 function. Basically
        #it takes a rho(Sn) as input, and returns its chi2-score given the error distr calculated from the input
        #self.uncdistr = self.make_rhouncdist()
        self.make_rhouncdist()
        def rho_chi2score(self, x):
            return -2*np.log(self.uncdistr(x))
        
        setattr(self, 'rho_chi2score', rho_chi2score.__get__(self, self.__class__))
    
    def low_levels_chi2(self, rholev, curr_nld):
        exp = curr_nld.y[((curr_nld.energies >= self.ExL1) & (curr_nld.energies <= self.ExL2))]
        err = curr_nld.yerr[(curr_nld.energies >= self.ExL1) & (curr_nld.energies <= self.ExL2)]
        theo = rholev[(rholev[:,0] >= self.ExL1) & (rholev[:,0] <= self.ExL2)][:,1]
        chi2_score = chisquared(theo, exp, err, DoF=1, method = 'linear', reduced=False)
        return chi2_score
        
    def dsigma_interp(self, xn):
        x = self.sigmamax - self.sigmamin
        y = self.dsigmamax - self.dsigmamin
        if y==0:
            return self.dsigmamin
        a = y/x
        yn = xn*a
        return self.dsigmamin + yn
    
    def import_low_Ex_from_file(self, path):
        self.nld_lvl = import_ocl(path, self.a0, self.a1, fermi=True)
    
    def NLD_from_countingdat(self, all_levels, Ex, FWHM):
        
        nld = 0
        bin_lowlim = Ex - 0.5
        bin_highlim = Ex + 0.5
        for el in all_levels:
            if bin_lowlim <= el <= bin_highlim:
                nld += 1
        
        ans = nld/abs(self.a1/1000) #TODO: ask somebody, why am I dividing by a1 here, actually? This comes from counting.c
        if FWHM > 0:
            pass #TODO: copy the ReadSmooth from counting.
        return ans 
    
    def import_low_Ex_from_function(self, countingdat_path, rhopaw_path, binsize = 1.0):
        all_levels = np.loadtxt(countingdat_path)/1000
        rhopaw = np.loadtxt(rhopaw_path)
        length = len(rhopaw)//2
        self.nld_lvl = np.zeros((length, 2))
        for i in range(length):
            self.nld_lvl[i,0] = self.a0 + self.a1*i
            self.nld_lvl[i,1] = self.NLD_from_countingdat(all_levels, self.nld_lvl[i,0], binsize)
    
    def write_input_cnt(self, rho, drho, D0, L1, L2, H1, H2, TL1, TL2, TH1, TH2, extr_model, Ex_low, s_low, sigma, FWHM):
        lines = []
        lines.append(['{:.6f}'.format(self.A), '1.847000', '{:.6f}'.format(self.Sn), '{:.6f}'.format(rho), '{:.6f}'.format(drho)])
        lines.append([str(int(L1)), str(int(L2)), str(int(H1)), str(int(H2))])
        lines.append([str(int(TL1)), str(int(TL2)), str(int(TH1)), str(int(TH2))])
        lines.append(['5', '18.280001', '-0.949000'])
        lines.append(['2', '0.562000', '-1.884088'])
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
        
        D0, L1, L2, H1, H2, TL1, TL2, TH1, TH2, extr_model, Ex_low, s_low, sigma, FWHM = inputlist
        if self.double:
            loc_dsigma = self.dsigma_interp(sigma)
        else:
            loc_dsigma = self.dsigma
        current_rho = D2rho(D0, self.target_spin, sigma)
        current_drho = drho(self.target_spin, sigma, loc_dsigma, D0, self.dD0, rho = current_rho)
        self.write_input_cnt(current_rho, current_drho, *inputlist)
        
        call([counting_code_path])
        
        curr_nld = nld('rhopaw.cnt',a0 = self.a0, a1 = self.a1, is_ocl = True)
        Anorm, alpha = import_Anorm_alpha('alpha.txt')
        curr_nld.L1 = L1
        curr_nld.L2 = L2
        curr_nld.H1 = H1
        curr_nld.H2 = H2
        curr_nld.TL1 = TL1
        curr_nld.TL2 = TL2
        curr_nld.TH1 = TH1
        curr_nld.TH2 = TH2
        curr_nld.extr_model = extr_model
        curr_nld.Ex_low = Ex_low
        curr_nld.s_low = s_low
        curr_nld.FWHM = FWHM
        curr_nld.Anorm = Anorm
        curr_nld.alpha = alpha
        curr_nld.rho = current_rho
        curr_nld.drho = current_drho
        curr_nld.spin_cutoff = sigma
        curr_nld.D0 = D0
        curr_nld.chi2 = self.low_levels_chi2(self.nld_lvl, curr_nld) + self.rho_chi2score(current_rho)
        return curr_nld

    def write_input_nrm(self, Gg, curr_nld):
        lines = []
        lines.append(['0', '{:.6f}'.format(self.Sn), '{:.6f}'.format(self.target_spin)])
        lines.append(['{:.6f}'.format(curr_nld.D0), '{:.6f}'.format(Gg)])
        lines.append(['105.000000', '150.000000'])
        
        newinput_nrm = ''
        for line in lines:
            curr_line = ' '
            for el in line:
                curr_line += (el + ' ')
            curr_line += '\n'
            newinput_nrm += curr_line
        
        with open('input.nrm', 'w') as write_obj:
            write_obj.write(newinput_nrm)

    def run_GSF_sim(self, current_Gg, curr_nld):
        
        self.write_input_nrm(current_Gg, curr_nld)
        
        call([normalization_code_path]);
        curr_gsf = gsf('strength.nrm', a0 = self.a0, a1 = self.a1, is_sigma = False, is_ocl = True)
        Bnorm = import_Bnorm('input.nrm')
        curr_gsf.L1 = curr_nld.L1
        curr_gsf.L2 = curr_nld.L2
        curr_gsf.H1 = curr_nld.H1
        curr_gsf.H2 = curr_nld.H2
        curr_gsf.TL1 = curr_nld.TL1
        curr_gsf.TL2 = curr_nld.TL2
        curr_gsf.TH1 = curr_nld.TH1
        curr_gsf.TH2 = curr_nld.TH2
        curr_gsf.extr_model = curr_nld.extr_model
        curr_gsf.Ex_low = curr_nld.Ex_low
        curr_gsf.s_low = curr_nld.s_low
        curr_gsf.FWHM = curr_nld.FWHM
        curr_gsf.Anorm = curr_nld.Anorm
        curr_gsf.Bnorm = Bnorm
        curr_gsf.alpha = curr_nld.alpha
        curr_gsf.Gg = current_Gg
        curr_gsf.rho = curr_nld.rho
        curr_gsf.drho = curr_nld.drho
        curr_gsf.chi2 = curr_nld.chi2 + ((self.Gg - curr_gsf.Gg)/self.dGg)**2
        curr_gsf.spin_cutoff = curr_nld.spin_cutoff
        curr_gsf.D0 = curr_nld.D0
        return curr_gsf
    
    def initialize_nld_inputlist(self):
        #Load the input list for counting.c with default values
        inputlist = [self.D0, self.L1, self.L2, self.H1, self.H2, self.TL1, self.TL2, self.TH1, self.TH2, self.extr_model, self.Ex_low, self.s_low, self.sigma, self.FWHM]
        return inputlist
    
    def MC_normalization(self, opt_range = [0.9,1.1], MC_range = 1000, load_lists = False):
        if load_lists:
            self.nlds = np.load('nlds.npy', allow_pickle = True)
            self.gsfs = np.load('gsfs.npy', allow_pickle = True)
        else:
            self.MC_range = MC_range
            
            '''
            #find limits where to pick random values of sigma2, D0, L1 and L2
            if self.double:
                self.sigma2_limits = [self.sigmamin - self.std_devs*self.dsigmamin, self.sigmamax + self.std_devs*self.dsigmamax]
            else:
                self.sigma2_limits = [self.sigma - self.std_devs*self.dsigma, self.sigma + self.std_devs*self.dsigma]
            self.D0_limits = [self.D0 - self.std_devs*self.dD0, self.D0 + self.std_devs*self.dD0]
            self.Gg_limits = [self.Gg - self.std_devs*self.dGg, self.Gg + self.std_devs*self.dGg]
            '''
            #initialize lists, start calculating NLDs and GSFs
            self.nlds = []
            self.gsfs = []
            os.chdir(self.OM_folderpath)
            
            #TODO: The "grid search" at the beginning should be another method, called "grid search" or something similar, that can be called independently, and that one can choose which parameter to vary, not only sigma
            if self.double and self.flat:
                sigma_lowlim = self.sigmamin*opt_range[0]
                sigma_highlim = self.sigmamax*opt_range[1]

                #initialize input list for NLD with default values
                nld_inputlist = self.initialize_nld_inputlist()
                #first, calculate the nlds and gsfs with "optimal conditions" (reccommended values for each parameter. Loop for 20 different spin-cutoff parameters)
                for s in np.linspace(sigma_lowlim, sigma_highlim, num = 20):
                    nld_inputlist[NLD_keys.index('sigma')] = s
                    curr_nld = self.run_NLD_sim(nld_inputlist)
                    curr_gsf = self.run_GSF_sim(self.Gg, curr_nld)
                    self.nlds.append(curr_nld)
                    self.gsfs.append(curr_gsf)
            
            #then, pick random nlds and gsfs
            progress_interval = self.MC_range // 10  # Print progress every 10%
            for i in range(MC_range):
                if (i + 1) % progress_interval == 0 or i + 1 == MC_range:
                    #Calculate progress percentage
                    progress = (i + 1) / MC_range * 100
                    #Print progress
                    print(f"Progress: {progress:.2f}% ({i+1}/{self.MC_range})\n", end="\r")
                
                
                #initialize input list for NLD with default values
                nld_inputlist = self.initialize_nld_inputlist()
                
                #find out if you need to vary the parameters. If so, change the default values from the list above.
                #start with L1, L2... These are ints, and L1 must be lower than L2.
                for couple in couples:
                    if (couple[0] in self.free_parameters) and (couple[1] in self.free_parameters) and (getattr(self, couple[0] + '_range') == getattr(self, couple[1] + '_range')):
                        par_range = getattr(self, couple[0] + '_range')
                        values = np.sort(rng.choice(np.arange(par_range[0], par_range[1] + 1), size = 2, replace = False))
                        nld_inputlist[NLD_keys.index(couple[0])] = values[0]
                        nld_inputlist[NLD_keys.index(couple[1])] = values[1]
                
                #extr_model can only be 1 or 2
                if 'extr_model' in self.free_parameters:
                    nld_inputlist[NLD_keys.index('extr_model')] = rng.randint(1,3)
                    
                #finally, the floats
                float_pars = ['D0', 'Ex_low', 's_low', 'sigma', 'FWHM']
                for float_par in float_pars:
                    if float_par in self.free_parameters:
                        par_range = getattr(self, float_par + '_range')
                        nld_inputlist[NLD_keys.index(float_par)] = rng.uniform(low = par_range[0], high = par_range[1])
                        
                #run the NLD simulation
                curr_nld = self.run_NLD_sim(nld_inputlist)
                self.nlds.append(curr_nld)
                
                if 'Gg' in self.free_parameters:
                    par_range = getattr(self, 'Gg_range')
                    for j in range(10):
                        current_Gg = rng.uniform(low = par_range[0], high = par_range[1])
                        curr_gsf = self.run_GSF_sim(current_Gg, curr_nld)
                        self.gsfs.append(curr_gsf)
                else:
                    curr_gsf = self.run_GSF_sim(self.Gg, curr_nld)
                    self.gsfs.append(curr_gsf)
                
            os.chdir(root_folder)
            
            np.save('nlds.npy', self.nlds)
            np.save('gsfs.npy', self.gsfs)
    
    def write_tables(self):
        best_fits = []
        valmatrices = []
        for i, (lst, lab) in enumerate(zip([self.nlds, self.gsfs], ['nld','gsf'])):
            valmatrix, best_fit = calc_errors_chis(lst, return_best_fit=True)
            valmatrix = clean_valmatrix(valmatrix)
            valmatrices.append(valmatrix)
            best_fits.append(best_fit)
            header = 'Energy [MeV], best_fit, best_fit-sigma, best_fit+sigma, staterr' 
            np.savetxt('data/generated/' + lab + '_whole.txt', valmatrix, header = header)
        self.best_fitting_nld = best_fits[0]
        self.best_fitting_gsf = best_fits[1]
        self.NLD_table = valmatrices[0]
        self.GSF_table = valmatrices[1]

        np.save('data/generated/best_fits.npy', best_fits)
        






a0 = -0.8433
a1 = 0.1221
A = 166
Sn = 6.243640
Ho166norm = normalization(rsc_folder, a0, a1, A, Sn)

sigma2FG = 5.546
sigma2RMI = 6.926
dsigma2FG = sigma2FG*0.00
dsigma2RMI = sigma2RMI*0.00
chi2_lim = [9,13]  #Fitting interval. Limits are included.
std_devs = 2.0

Ho166norm.set_attributes(sigma = [sigma2FG, sigma2RMI], 
                         dsigma = [dsigma2FG, dsigma2RMI], 
                         D0 = 4.35, 
                         dD0 = 0.15, 
                         target_spin = 7/2, 
                         Ex1 = a0 + a1*chi2_lim[0],
                         Ex2 = a0 + a1*chi2_lim[1],
                         Ex_low = 0.2,
                         s_low = 4.5,
                         H1 = 32,
                         H2 = 45,
                         TL1 = 18,
                         TL2 = 19,
                         TH1 = 41,
                         TH2 = 45,
                         Gg = 84.0,
                         dGg = 5.0,
                         extr_model = 2,
                         FWHM = 150.0
                         )

Ho166norm.set_variation_intervals(std_devs,
                       D0 = ['err', 'err'],
                       L1 = [8,14],
                       L2 = [8,14],
                       sigma = ['err', 'err'],
                       Gg = ['err', 'err'],
                       flat = True)

Ho166norm.import_low_Ex_from_file('data/rholev.cnt')
#Ho166norm.import_low_Ex_from_function('rhosigchi/counting.dat', 'rhosigchi/rhopaw.rsg')
Ho166norm.make_rho_chi2score()
Ho166norm.MC_normalization(opt_range = [0.9,1.1], MC_range = 100)
Ho166norm.write_tables()


















        