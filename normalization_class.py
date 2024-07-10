#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Normalization class, used for the normalization procedure of the Oslo method,
it uses Monte Carlo simulations to propagate the systematic errors from the input
data to the final results, for customized error PDFs. This is a systematized, 
effectivized and expanded version of the algorithm used in https://doi.org/10.1103/PhysRevC.107.034605
'''


import numpy as np
from functions import D2rho, drho, chisquared, import_Anorm_alpha, import_Bnorm, calc_errors_chis, clean_valmatrix
import matplotlib.pyplot as plt
from subprocess import call
import os
from scipy.special import erf
from nld_gsf_classes import import_ocl, nld, gsf
rng = np.random.default_rng()#seed=1070

#full address of modified counting and normalization codes
counting_code_path = "/home/francesco/oslo-method-software-auto/prog/counting"
normalization_code_path = "/home/francesco/oslo-method-software-auto/prog/normalization"

#working directory
root_folder = os.getcwd()

#some useful lists
NLD_keys = ['D0', 'L1', 'L2', 'H1', 'H2', 'TL1', 'TL2', 'TH1', 'TH2', 'extr_model', 'Ex_low', 's_low', 'sigma', 'FWHM']
couples = [['L1', 'L2'],
           ['H1', 'H2'],
           ['TL1', 'TL2'],
           ['TH1', 'TH2']]


#normalization class
class normalization:
    def __init__(self, OM_folderpath, A, Sn):
        #self.a0 = a0
        #self.a1 = a1
        self.OM_folderpath = OM_folderpath
        self.A = A
        self.Sn = Sn
        
        #read some important data from rhosigchi output
        rhosp = np.genfromtxt(self.OM_folderpath + '/rhosp.rsg', skip_header=6, max_rows = 1, delimiter =',')
        self.a0 = rhosp[1]/1000
        self.a1 = rhosp[2]/1000
        rhopaw_path = self.OM_folderpath + '/rhopaw.rsg'
        self.rhopaw = import_ocl(rhopaw_path, self.a0, self.a1)
        sigpaw_path = self.OM_folderpath + '/sigpaw.rsg'
        self.sigpaw = import_ocl(sigpaw_path, self.a0, self.a1)
        self.rhopaw_length = len(self.rhopaw[:,0])
        self.sigpaw_length = len(self.sigpaw[:,0])
        self.dim = min(self.rhopaw_length, self.sigpaw_length)
        
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
        
        Ex_low: the energy where the spin_cutoff value for low excitation energy is evaluated (used for the linear fit, ALEX method in counting.c. see https://doi.org/10.1140/epja/i2015-15170-4 (or the application in https://doi.org/10.1103/PhysRevC.107.034605))
        s_low: the corresponding spin_cutoff parameter at Ex_low
        
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
        '''
        
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
        
        nld = 0
        bin_lowlim = Ex - binsize/2
        bin_highlim = Ex + binsize/2
        for el in all_levels:
            if bin_lowlim <= el < bin_highlim:
                nld += 1
        
        ans = nld/abs(self.a1)
        
        return ans 
    
    def import_low_Ex_nld_raw(self, countingdat_path = None, binsize = 1.0):
        '''
        This will use the first (naive) algorithm from counting.c to read the low Ex
        level density from counting.dat. This is only used to guess L1 and L2.
        '''
        
        if countingdat_path == None:
            countingdat_path = self.OM_folderpath + '/counting.dat'
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
            countingdat_path = self.OM_folderpath + '/counting.dat'
        
        all_levels = np.loadtxt(countingdat_path)/1000
        self.nld_lvl = np.zeros((self.dim, 2))
        
        
        if self.FWHM > 0:
            
            sigma = self.FWHM/(2*np.sqrt(2*np.log(2)))/1000
            
            for i in range(self.dim):
                self.nld_lvl[i,0] = self.a0 + self.a1*i
                #self.nld_lvl[i,1] = 0
            
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
    
    def write_input_cnt(self, rho, drho, D0, L1, L2, H1, H2, TL1, TL2, TH1, TH2, extr_model, Ex_low, s_low, sigma, FWHM):
        '''
        Writes the counting.c input file
        '''
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
        self.write_input_cnt(current_rho, current_drho, *inputlist)
        #TODO: the following line runs the modified counting.c script. It takes the newly written input file input.cnt and it writes all the normal output files.
        call([counting_code_path])
        #TODO: ...these files are then read and stored in an instantiation of the nld class. It would be nice in the future if the program was purely python, and "counting" was just a python function. Potentially, something Ompy already has.
        curr_nld = nld('rhopaw.cnt', a0 = self.a0, a1 = self.a1)
        curr_nld.clean_nans()
        curr_nld.add_rhoSn(self.Sn, current_rho, current_drho)
        Anorm, alpha = import_Anorm_alpha('alpha.txt')
        curr_nld.L1 = vardict['L1'] #L1
        curr_nld.L2 = vardict['L2']
        curr_nld.H1 = vardict['H1']
        curr_nld.H2 = vardict['H2']
        curr_nld.TL1 = vardict['TL1']
        curr_nld.TL2 = vardict['TL2']
        curr_nld.TH1 = vardict['TH1']
        curr_nld.TH2 = vardict['TH2']
        curr_nld.extr_model = vardict['extr_model']
        curr_nld.Ex_low = vardict['Ex_low']
        curr_nld.s_low = vardict['s_low']
        curr_nld.FWHM = vardict['FWHM']
        curr_nld.Anorm = Anorm
        curr_nld.alpha = alpha
        curr_nld.rho = current_rho
        curr_nld.drho = current_drho
        curr_nld.spin_cutoff = vardict['sigma']
        curr_nld.D0 = vardict['D0']
        extr_mat = import_ocl('fermigas.cnt', self.a0, self.a1, no_errcol = True)
        curr_nld.extr_mat = extr_mat
        curr_nld.rhoSn_chi2 = self.rho_chi2score([vardict[variable] for variable in self.rho_variables])
        curr_nld.chi2 = self.low_levels_chi2(self.nld_lvl, curr_nld) + curr_nld.rhoSn_chi2
        #self.rho_chi2score([vardict['sigma'], vardict['D0']])
        return curr_nld

    def write_input_nrm(self, Gg, curr_nld):
        '''
        Writes the normalization.c input file
        '''
        lines = []
        lines.append(['0', '{:.6f}'.format(self.Sn), '{:.6f}'.format(self.target_spin)])
        lines.append(['{:.6f}'.format(curr_nld.D0), '{:.6f}'.format(Gg)])
        lines.append(['105.000000', '{:.6f}'.format(self.FWHM)])
        
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
        '''
        runs the normalization.c program, and reads and save the results in a gsf object,
        and returns the instance
        '''
        self.write_input_nrm(current_Gg, curr_nld)
        
        #TODO: the following line runs the modified normalization.c script. It takes the newly written input file input.nrm and it writes all the normal output files.
        call([normalization_code_path]);
        #TODO: ...these files are then read and stored in an instantiation of the gsf class. It would be nice in the future if the program was purely python, and "normalization" was just a python function. Potentially, something Ompy already has.
        curr_gsf = gsf('strength.nrm', a0 = self.a0, a1 = self.a1)
        curr_gsf.clean_nans()
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
        '''
        Load the input list for counting.c with default values
        '''
        inputlist = [self.D0, self.L1, self.L2, self.H1, self.H2, self.TL1, self.TL2, self.TH1, self.TH2, self.extr_model, self.Ex_low, self.s_low, self.sigma, self.FWHM]
        return inputlist
    
    def prompt_grid_search(self, variable, lowlim, highlim, num):
        '''
        runs the small grid search at the beginning
        '''
        nld_inputlist = self.initialize_nld_inputlist()
        for s in np.linspace(lowlim, highlim, num = num):
            nld_inputlist[NLD_keys.index(variable)] = s
            curr_nld = self.run_NLD_sim(nld_inputlist)
            curr_gsf = self.run_GSF_sim(self.Gg, curr_nld)
            self.nlds.append(curr_nld)
            self.gsfs.append(curr_gsf)
    
    def MC_normalization(self, opt_range = [0.9,1.1], MC_range = 1000, load_lists = False, num = 20):
        
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
            os.chdir(self.OM_folderpath)
            
            #first: a small grid search by varying only one of the rho variables at a time.
            self.grid_searches = 0
            self.num = num
            for variable in self.rho_variables:
                vardouble = getattr(self, variable + 'double')
                varflat = getattr(self, variable + 'flat')
                
                if vardouble and varflat:
                    lowlim = getattr(self, variable + 'min')*opt_range[0]
                    highlim = getattr(self, variable + 'max')*opt_range[1]
                    self.prompt_grid_search(variable, lowlim, highlim, num)
                    self.grid_searches += 1
            
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
                current_Gg = self.Gg
                
                #find out if you need to vary the parameters. If so, change the default values from the list above.
                #start with L1, L2... These are ints, and L1 must be lower than L2.
                for couple in couples:
                    if (couple[0] in self.free_parameters) and (couple[1] in self.free_parameters) and (getattr(self, couple[0] + '_range') == getattr(self, couple[1] + '_range')):
                        par_range = getattr(self, couple[0] + '_range')
                        values = np.sort(rng.choice(np.arange(par_range[0], par_range[1] + 1), size = 2, replace = False))
                        nld_inputlist[NLD_keys.index(couple[0])] = values[0]
                        nld_inputlist[NLD_keys.index(couple[1])] = values[1]
                
                #extr_model can only be 1 or 2. If this is a free parameter, 1 or 2 will be picked at random.
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
                    #for each NLD simulation, run 10 GSF simulations (it seems like I need it to get enough simulations, and the most effective way instead of incresing MC_range)
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
    
    def write_tables(self, graphic_function = 'find_chis', path = ''):
        '''
        find the errors from the simulations, and write useful tables
        '''
        best_fits = []
        valmatrices = []
        for i, (lst, lab) in enumerate(zip([self.nlds, self.gsfs], ['nld','gsf'])):
            valmatrix, best_fit = calc_errors_chis(lst, lab = lab, return_best_fit=True, graphic_function = graphic_function)
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
        
        scaling_factor = 0.9
        fig0, ax0 = plt.subplots(figsize = (5.0*scaling_factor, 3.75*scaling_factor), dpi = 300)
        fig1, ax1 = plt.subplots(figsize = (5.0*scaling_factor, 3.75*scaling_factor), dpi = 300)
        #ax0.plot(np.zeros(1), np.zeros([1,5]), color='w', alpha=0, label=' ')
        
        Sn_lowerr = self.NLD_table[-1,1] - self.NLD_table[-1,2]
        Sn_higherr = self.NLD_table[-1,3] - self.NLD_table[-1,1]
        ax0.errorbar(self.Sn, self.NLD_table[-1,1], yerr=np.array([[Sn_lowerr, Sn_higherr]]).T, ecolor='g',linestyle=None, elinewidth = 4, capsize = 5, label=r'$\rho$ at Sn')
        
        ax0.fill_between(self.NLD_table[:-1,0], self.NLD_table[:-1,3], self.NLD_table[:-1,-3], color = 'b', alpha = 0.2, label=r'1$\sigma$ conf.')
        ax0.plot(self.nld_lvl[:,0], self.nld_lvl[:,1], 'k-', label='Known lvs.')
        ax0.axvspan(self.ExL1, self.ExL2, alpha=0.2, color='red',label='Fitting intv.')
        ax0.errorbar(self.NLD_table[:-1,0], self.NLD_table[:-1,1],yerr=self.NLD_table[:-1,-1], fmt = '.', color = 'b', ecolor='b', label='This work')
        ax0.plot(self.best_fitting_nld.extr_mat[(self.H1-5):,0], self.best_fitting_nld.extr_mat[(self.H1-5):,1], color = 'k', linestyle = '--', alpha = 1, label = 'extrap.')
        
        ax1.fill_between(self.GSF_table[:,0], self.GSF_table[:,3], self.GSF_table[:,-3], color = 'b', alpha = 0.2, label=r'1$\sigma$ conf.')
        ax1.errorbar(self.GSF_table[:,0], self.GSF_table[:,1],yerr=self.GSF_table[:,-1], fmt = '.', color = 'b', ecolor='b', label='This work')
        
        #Plot experiment data
        for ax in [ax0,ax1]:
            ax.set_yscale('log')
            ax.legend(frameon = False)
        ax0.set_xlabel(r'$E_x$ [MeV]')
        ax1.set_xlabel(r'$E_\gamma$ [MeV]')
        ax0.set_ylabel(r'NLD [MeV$^{-1}$]')
        ax1.set_ylabel(r'GSF [MeV$^{-3}$]')
        ax0.set_ylim([min(self.NLD_table[:-1,-3])*0.5, self.NLD_table[-1,3]*2])
        ax0.set_xlim([self.NLD_table[0,0]-0.5, self.NLD_table[-1,0]+0.5])
        fig0.gca().set_xlim(left=min(self.NLD_table[:-1,0])-0.5)
        fig0.tight_layout()
        fig1.tight_layout()
        fig0.show()
        fig1.show()
        if save:
            fig0.savefig('nld.pdf', format = 'pdf')
            fig1.savefig('gsf.pdf', format = 'pdf')

















        