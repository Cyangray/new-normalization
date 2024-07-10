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

'''
1) one creates an instance of the normalization class. This is done in the
following three lines.
'''
A = 166 #mass number
Sn = 6.243640 #MeV
rhosigchi_folder = 'rhosigchi'
Ho166norm = normalization(rhosigchi_folder, A, Sn)

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
                         TH1 = 41,         # if you skip some, they will be guessed with the same algorithm as in counting.c. Try e.g. to comment out TH1 and TH2.
                         TH2 = 45,
                         s_low = 4.5,       # s_low and Ex_low are the average spin and its excitation energy at low Ex, from which one can run a linear fit to sigma at rho(Sn). (the ALEX method in counting.c. See ***add Guttormsen ref***)
                         Ex_low = 0.2,
                         Gg = 84.0,         # average partial γ-decay width (meV)
                         dGg = 5.0,         # error in Gg
                         extr_model = 2,    # model used to extrapolate the nld to rho(Sn). 1 for constant temperature, 2 for Fermi gas. 
                         FWHM = 150.0       # Full-width half-maximum
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
                       sigmaflat = True,        # default is sigmaflat = False. This tells how to treat the uncertainty within the spin-cutoff values, if 2 are given in "set attributes". sigmaflat = True assumes all values within the two given as equally probable, and outside this range, the respective dsigma standard deviations will apply. If sigmaflat = False, the two sigma values given in "set_attributes" will be considered as the lower and upper error, and an average will be calculated to be the most probable. See ***Ho paper***
                       D0flat = False)           # the same as for sigmaflat, but this time for D0.

'''
4) if you have put your counting.dat file in the rhosigchi folder, import the data with this method
'''
Ho166norm.import_low_Ex_nld()

'''
5) run the MC simulations. This might take a minute or two
'''
opt_range = [0.9,1.1] # before running the MC simulation, the algorithm does a quick, mini-gridsearch with the most probable values. opt_range tells how far away from the suggested values one should stride.
MC_range = 1000       # the number of MC simulations
Ho166norm.MC_normalization(opt_range = opt_range, MC_range = MC_range, load_lists = False)

'''
6) translates the results into readable tables. function = 'find_chis_interp' is a bit slower, but more accurate.
'''
Ho166norm.write_tables(graphic_function = 'find_chis_interp', path = '')
#Ho166norm.write_tables(graphic_function = 'find_chis', path = '')

'''
7) plot graphs.
'''
Ho166norm.plot_graphs()
    
