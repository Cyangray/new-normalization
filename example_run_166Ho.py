#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Guide:
This is an example of usage of the normalization code, which takes as input the
output of rhosigchi (in the form of the path to the folder where rhosigchi took
place) and returns (among other things) two tables, one for NLD and one for GSF
including the reccomended values, together with the lower and upper systematic
error, and the statistical error.
'''

from normalization_class import normalization
import numpy as np
from functions import SLO_arglist, GLO_hybrid_arglist
import time

'''
1) one creates an instance of the normalization class. This is done in the
following lines.
'''
A = 166 #mass number
Z = 67
Sn = 6.243640 #MeV
rhosigchi_folder = 'rhosigchi'
oslo_method_software_path = '/home/francesco/oslo-method-software-auto'
Ho166norm = normalization(rhosigchi_folder, oslo_method_software_path, A, Z, Sn)

'''
2) one provides the class with the information for the normalization with "set_attributes".
'''
sigma2FG = 5.546
sigma2RMI = 6.926
sigma = [sigma2FG, sigma2RMI]
dsigma2FG = sigma2FG*0.01
dsigma2RMI = sigma2RMI*0.01
dsigma = [dsigma2FG, dsigma2RMI]

Ho166norm.set_attributes(sigma = sigma,     # spin-cutoff parameter. float or list/touple
                         dsigma = dsigma,   # error in sigma. Same type as sigma
                         D0 = 4.35,         # neutron average resonance spacing (in eV). float or list/touple
                         dD0 = 0.15,        # error in D0. Same type as D0.
                         target_spin = 7/2, # spin of the target nucleus (Z, N-1)
                         L1 = 9,            # lower limit of where to fit the nld to the known levels at low Ex. Can be given as a bin
                         #ExL1 = 0.2556,    # ...or can be given as an energy in MeV. This will be automatically translated to the closest bin
                         L2 = 13,           # upper limit of where to fit the nld to the known levels at low Ex. Can again be given as a bin or an energy, independently of how L1 was given, but please provide both the upper or lower limit, otherwise both will be guessed
                         H1 = 32,           # The same is true for all L1, L2, H1, H2 etc
                         H2 = 45,
                         TL1 = 18,
                         TL2 = 19,
                         TH1 = 41,          # if you skip some, they will be guessed with the same algorithm as in counting.c. Try e.g. to comment out TH1 and TH2.
                         TH2 = 45,
                         s_low = 4.5,       # s_low and Ex_low are the average spin and its excitation energy at low Ex, from which one can run a linear fit to sigma at rho(Sn). (the ALEX method in counting.c. See ***add Guttormsen ref***)
                         Ex_low = 0.2,
                         Gg = 84.0,         # average partial γ-decay width (meV)
                         dGg = 5.0,         # error in Gg
                         extr_model = 2,    # model used to extrapolate the nld to rho(Sn). 1 for constant temperature, 2 for Fermi gas. 
                         FWHM = 150.0,      # Full-width half-maximum
                         a = 18.280001,     # a parameter for the Fermi gas etrapolation
                         #da = 18.280001*0.01, # do you want to add an uncertainty to the rho extrapolation parameter? You can do it. just add "d" in front of the name of the parameter, and put the value of the error. If you want to propagate this unertainty, remember to add, in this case, "a = ['err', 'err']" in set_variation_intervals below
                         E1 = -0.949000    # E1 parameter
                         )

'''
3) we pick the variables we'd like to vary in our MC simulation. Use the same keywords as in "set attributes".
'''
std_devs = 2.0
Ho166norm.set_variation_intervals(std_devs,     # std_devs: how many standard deviations the parameter will be varied in the MC simulation.
                       D0 = ['err', 'err'],     # if you put ['err', 'err'], then the dD0 value will be used up and down from the given value(s) in "set_attributes". The same is true for all other variables (like e.g. sigma or Gg)
                       L1 = [8,14],             # while the interval in "set_attributes" indicates where the chi2 test will be run, this interval gives the range where the MC algorithm will pick the L1 and L2 values. L2 is always bigger than L1. See ***my Sb and Ho articles*** for more explanations on what's going on here (even though back then I used a grid search and not a MC simulation)
                       L2 = [8,14],             # this can be the same as for L1. L2 will be always randomly picked to be bigger than L1. If L2 is not provided here, the value given (or guessed) in "set_attributes" will be always used. In this case, make sure that L2 is bigger than the L1 range given here.
                       #TL1 = [16,22],           # vary TL1 and TL2 in the same ways as L1 and L2. The same can be done with all bin intervals.
                       #TL2 = [16,22],
                       sigma = ['err', 'err'],
                       Gg = ['err', 'err'],
                       #a = ['err', 'err'],
                       sigmaflat = True,        # default is sigmaflat = False. This tells how to treat the uncertainty within the spin-cutoff values, if 2 are given in "set attributes". sigmaflat = True assumes all values within the two given as equally probable, and outside this range, the respective dsigma standard deviations will apply. If sigmaflat = False, the two sigma values given in "set_attributes" will be considered as the lower and upper error, and an average will be calculated to be the most probable. See ***Ho paper***
                       D0flat = False)           # the same as for sigmaflat, but this time for D0.

'''
4) if you have put your counting.dat file in the rhosigchi folder, import the data with this method
'''
Ho166norm.import_low_Ex_nld()

'''
5) run the MC simulations. This might take a minute or two
    load_lists = True, if you already have run the program once, and you want to load the data from the saved files instead of calculating it once more
'''
opt_range = [0.9,1.1] # before running the MC simulation, the algorithm does a quick, mini-gridsearch with the most probable values. opt_range tells how far away from the suggested values one should stride.
MC_range = 1000       # the number of MC simulations
Ho166norm.MC_normalization(opt_range = opt_range, MC_range = MC_range, N_cores = 8, load_lists = False)

'''
6) translates the results into readable tables.
'''
Ho166norm.write_NLD_GSF_tables(path = '')

'''
7) calculate NLD and GSF models with TALYS (if you want to plot them together with the results)
    load_lists = True, if you already have run the program once, and you want to load the data from the saved files instead of calculating it once more
'''
talys_root_path = '/home/francesco/talys' #put your own path to the TALYS root folder here
talys_executable_path = talys_root_path + '/bin/talys' #for example
talys_version = '2.00' #either '1.96' or '2.00'
Ho166norm.set_TALYS_version(talys_root_path, talys_executable_path, talys_version)
#Ho166norm.calc_TALYS_models(load_lists = False, N_cores = 8, number_of_strength_models = 9)

'''
8) plot graphs.
'''
Ho166norm.plot_graphs()
"""
'''
9) Run TALYS many times to calculate the ncrates and MACSs.
9a) First, you have to define "high_energy_extrap": a (2,n) array with energies and gsf for higher energies. For example, this could be the experimental GDR data for a nearby nucleus, or an exponential extrapolation
    Second, you have to define "M1pars": this could be a float between 0 and 1, a list 3 elements long, or a list 6 elements long.
        if it's a float, the M1 strength will be simply calculated as a fraction of the total strength: i.e. if M1 = 0.1, then for every bin the strength will be divided between 10% M1 and 90% E1
        if it's a list of 3, these will be the centroid, the Gamma and the sigma of an SLO (e.g. of the spin-flip resonance)
        if it's a list of 6, these will be the centroid, the Gamma and the sigma of two SLOs (e.g., the spin-flip and the scissors resonances)
'''
high_energies = np.linspace(0.1, 20, 1000)
x_values_cont = np.linspace(0.1, 20, 1000)
GDR1 = [0.59, 12.40,3.50,323]
GDR2 = [0.59, 14.80,1.82,183]
PDR = [6.07,1.89,5.0]
M1pars = [3.20,1.00,0.40]
SR = [3.20,1.00,0.40]
PDR7 = [6.07,1.89,5.0/7]
M1pars.extend(PDR7)
high_Ho165_vals = GLO_hybrid_arglist(x_values_cont, GDR1) + GLO_hybrid_arglist(x_values_cont, GDR2) + SLO_arglist(x_values_cont, PDR) + SLO_arglist(x_values_cont, SR)
indexes_to_delete = np.argwhere(high_energies<6.5) #delete values less than 6.5 MeV
Ho165energies = np.delete(high_energies, indexes_to_delete, 0)
Ho165y = np.delete(high_Ho165_vals, indexes_to_delete, 0)
high_energy_extrap = np.c_[Ho165energies, Ho165y]

'''
9b) Here you actually calculate ncrates and MACSs with TALYS
    chi2_window = a bit difficult to explain, but the recommended value is 1.0, or as high as possible, the maximum is 2.0. 
        The higher it is, the more simulations are run, and the longer time it takes, but the result is more precise. 
        Run once at e.g. chi2_window = 0.2 to get a rough result if you want something quick. If it gives an error, you have to increase the number.
    load_lists = True, if you already have run the program once, and you want to load the data from the saved files instead of calculating it once more
'''
start_whole = time.time()
Ho166norm.run_TALYS_sims_parallel(M1pars, high_energy_extrap, chi2_window = 1.0, N_cores = 8, load_lists = False, label = '')
finish_whole = time.time()
print(f'Whole parallel TALYS simulation ended in {finish_whole - start_whole} seconds')
'''
10) write results into human readable tables
'''
Ho166norm.write_ncrate_MACS_tables(load_lists = False, label = '')
"""