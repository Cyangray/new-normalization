#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:47:44 2024

@author: francesco
"""

from numba import njit
import numpy as np

@njit
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

@njit
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

@njit
def rhofg(ex, a, T, E1, E0, imodel, b1, b2):
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

@njit
def _corrL(a0, a1, nld_lvl, rhopaw, L1, L2, alpha, Anorm):
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
            Ex = a0 + a1*i
            rhoL = max(nld_lvl[i, 1], 0.01)
            cc = corr * Anorm * np.exp(alpha*Ex) * rhopaw[i, 1]
            dc2 = (corr * Anorm * np.exp(alpha*Ex) * rhopaw[i, 2])**2
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

@njit
def _corrH(a0, a1, rhopaw, H1, H2, alpha, Anorm, a, T, E1, E0, extr_model, b1, b2, eta):
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
            Ex = a0 + a1*i
            cc = corr * Anorm * np.exp(alpha*Ex) * rhopaw[i, 1]
            dc2 = (corr * Anorm * np.exp(alpha*Ex) * rhopaw[i, 2])**2
            _, rhox = rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)
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

@njit
def _temperature(a0, a1, FGmax, Ex, rhotmopaw):
    for i in range(FGmax):
        if a0 + a1*i > Ex:
            break
    i2 = i
    i1 = i2-1
    num = 0
    temp = 0
    temp0 = 0
    tempL = 0
    tempH = 0
    if (rhotmopaw[i1] > 0) and (rhotmopaw[i2] > 0) and (rhotmopaw[i2] > rhotmopaw[i1]):
        temp0 = a1/(np.log(rhotmopaw[i2]) - np.log(rhotmopaw[i1]))
        num = num + 1
    if (i1 > 0) and (rhotmopaw[i1 - 1] > 0) and (rhotmopaw[i2 - 1] > 0) and (rhotmopaw[i2 - 1] > rhotmopaw[i1 - 1]):
        tempL = a1/(np.log(rhotmopaw[i2-1]) - np.log(rhotmopaw[i1-1]))
        num = num + 1
    if (i2 < 8192-1) and (rhotmopaw[i1 + 1] > 0) and (rhotmopaw[i2 + 1] > 0) and (rhotmopaw[i2 + 1] > rhotmopaw[i1 + 1]):
        tempH = a1/(np.log(rhotmopaw[i2+1]) - np.log(rhotmopaw[i1+1]))
        num = num + 1
    if num > 0:
        temp = (tempL + temp0 + tempH)/num
    return temp

@njit
def _spin_distribution(a0, a1, FGmax, Ex, I, sigma2):
    for i in range(FGmax):
        if a0 + a1*i > Ex:
            break
    i2 = i
    i1 = i2-1
    s2 = sigma2[i1] + ((sigma2[i2] - sigma2[i1])/a1) * (Ex - (a0 + a1*i1) )
    s2 = max(s2, 1.0)
    g_Ex = (2*I + 1)*np.exp(-(I+0.5)**2/(2*s2)) / (2*s2)
    if g_Ex > 1e-20:
        return g_Ex
    else:
        return 0.

@njit
def _extendL(a0, a1, TL1, TL2, sigdimBn, nsig):
    nsigL = np.zeros((8192,3))
    steps = 1000
    abest = 0
    bbest = 0
    x1 = a0 + a1*TL1
    x2 = a0 + a1*TL2
    y1 = np.log(nsig[TL1, 1])
    y2 = np.log(nsig[TL2, 1]) 
    if TL2 - TL1 > 3:
        x1 = a0 + a1*TL1 + 0.5
        x2 = a0 + a1*TL2 + 0.5
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
                x = a0 + a1*k
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
        x = a0 + a1*i
        nsigL[i,1] = np.exp(abest * x + bbest)
        if i < TL1:
            nsigL[i,1] = x**3 * np.exp(abestL * x + bbestL)
    return nsigL

@njit
def _extendH(a0, a1, TH1, TH2, FGmax, nsig):
    nsigH = np.zeros((8192,3))
    steps = 1000
    abest = 0
    bbest = 0
    x1 = a0 + a1*TH1
    x2 = a0 + a1*TH2
    y1 = np.log(nsig[TH1, 1])
    y2 = np.log(nsig[TH2, 1]) 
    if TH2 - TH1 > 3:
        x1 = a0 + a1*TH1 + 0.5
        x2 = a0 + a1*TH2 + 0.5
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
                x = a0 + a1*k
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
        x = a0 + a1*i
        nsigH[i,1] = np.exp(abestH * x + bbestH)
    return nsigH

@njit
def _rhoexp(a0, a1, FGmax, Ex, rhotmopaw):
    for i in range(FGmax):
        if ((a0 + a1*i)/1000) > Ex:
            break
    i2 = i
    i1 = i2-1
    rhox = rhotmopaw[i1] + ((rhotmopaw[i2]-rhotmopaw[i1])/a1)*(Ex-(a0 + a1*i1))
    return rhox

@njit
def _T_eg(a0, a1, NchBn, eg, sigpawext):
    i2 = -1
    for ii in range(NchBn):
        if eg > (a0 + a1*ii)*1000:
            i2 = ii + 1
    if i2 == -1 or i2 > NchBn-1:
        i2 = NchBn - 1
    if i2 == 0:
        i2 = 1
    i1 = i2 - 1
    eg1 = (a0 + a1*i1)*1000
    eg2 = (a0 + a1*i2)*1000
    Teg = sigpawext[i1] + (sigpawext[i2] - sigpawext[i1])*(eg-eg1)/(eg2-eg1)
    Teg = max(Teg, 1e-10)
    return Teg

@njit
def _rho_ex(a0, a1, NchBn, ex, rhotmopaw):
    i2 = -1
    for ii in range(NchBn):
        if ex > (a0 + a1*ii)*1000:
            i2 = ii + 1
    if i2 == -1 or i2 > NchBn-1:
        i2 = NchBn - 1
    if i2 == 0:
        i2 = 1
    i1 = i2 - 1
    ex1 = (a0 + a1*i1)*1000
    ex2 = (a0 + a1*i2)*1000
    rhox = rhotmopaw[i1] + (rhotmopaw[i2] - rhotmopaw[i1])*(ex - ex1)/(ex2 - ex1)
    rhox = max(rhox, 0)
    return rhox

@njit
def _sig_ex(a0, a1, NchBn, ex, spincut):
    i2 = -1
    for ii in range(NchBn):
        if ex > (a0 + a1*ii)*1000:
            i2 = ii + 1
    if i2 == -1 or i2 > NchBn-1:
        i2 = NchBn - 1
    if i2 == 0:
        i2 = 1
    i1 = i2 - 1
    ex1 = (a0 + a1*i1)*1000
    ex2 = (a0 + a1*i2)*1000
    sigx = 2*spincut[i1]**2 + (2*spincut[i2]**2 - 2*spincut[i1]**2)*(ex - ex1)/(ex2 - ex1)
    sigx = max(sigx, 0.01)
    return sigx

@njit
def _normalization_integral(a0, a1, NchBn, Sn, target_spin, sigpawext, rhotmopaw, spincut):
    
    '''
    runs the normalization.c integral, now translated in Python,
    and returns the integral result
    '''
    Eres = 400.
    de = 10.
    Bn_keV = Sn*1000
    It = target_spin
    eg = 0
    
    int1 = 0
    int2 = 0
    int3 = 0
    int4 = 0
    if It == 0.0:
        while eg < Bn_keV + Eres:
            ex = Bn_keV - eg
            Teg = _T_eg(a0, a1, NchBn, eg, sigpawext)
            rhoex = _rho_ex(a0, a1, NchBn, ex, rhotmopaw)
            sigex = _sig_ex(a0, a1, NchBn, ex, spincut)
            int3 += Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
            int4 += Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
            eg += de
        Int = int3 + int4
        
    elif It == 0.5:
        while eg < Bn_keV + Eres:
            ex = Bn_keV - eg
            Teg = _T_eg(a0, a1, NchBn, eg, sigpawext)
            rhoex = _rho_ex(a0, a1, NchBn, ex, rhotmopaw)
            sigex = _sig_ex(a0, a1, NchBn, ex, spincut)
            int2 += Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
            int3 += Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
            int4 += Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
            eg += de
        Int = int2 + 2*int3 + int4
        
    elif It == 1.0:
        while eg < Bn_keV + Eres:
            ex = Bn_keV - eg
            Teg = _T_eg(a0, a1, NchBn, eg, sigpawext)
            rhoex = _rho_ex(a0, a1, NchBn, ex, rhotmopaw)
            sigex = _sig_ex(a0, a1, NchBn, ex, spincut)
            int2 += Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
            int3 += Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
            int4 += Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
            eg += de
        Int = 2*int2 + 2*int3 + int4

    elif It > 1.0:
        while eg < Bn_keV + Eres:
            ex = Bn_keV - eg
            Teg = _T_eg(a0, a1, NchBn, eg, sigpawext)
            rhoex = _rho_ex(a0, a1, NchBn, ex, rhotmopaw)
            sigex = _sig_ex(a0, a1, NchBn, ex, spincut)
            int1 += Teg*rhoex * (It / sigex) * np.exp(-(It - 1)**2 / sigex)
            int2 += Teg*rhoex * (It / sigex) * np.exp(-(It - 0)**2 / sigex)
            int3 += Teg*rhoex * (It / sigex) * np.exp(-(It + 1)**2 / sigex)
            int4 += Teg*rhoex * (It / sigex) * np.exp(-(It + 2)**2 / sigex)
            eg += de
        Int = int1 + 2*int2 + 2*int3 + int4
    
    return Int

@njit
def _nld_talys(a0, a1, FGmax, rhotmopaw, sigma2, start_spin):
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
        nld_calc[i] = _rhoexp(a0, a1, FGmax, energies[i], rhotmopaw)
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
        talys_nld_cnt[i,1] = _temperature(a0, a1, FGmax, energies[i], rhotmopaw)
        talys_nld_cnt[i,2] = n_cum
        talys_nld_cnt[i,3] = nld_calc[i]
        talys_nld_cnt[i,4] = nld_calc[i]
        for j in range(stop_spin):
            I = j + start_spin
            x = nld_calc[i]*_spin_distribution(a0, a1, FGmax, energies[i], I, sigma2)
            talys_nld_cnt[i,4+j] = x
    return talys_nld_cnt

@njit
def _counting(a0, a1, FGmax, dim, current_rho, current_drho, D0, L1, L2, H1, H2, TL1, TL2, TH1, TH2, extr_model, Ex_low, s_low, sigma, FWHM, a, E1, T, E0, Sn, rhopaw, sigpaw, nld_lvl):
    
    b2 = (sigma**2 - s_low**2)/(Sn - Ex_low)
    b1 = s_low**2 - b2*Ex_low;
    
    _, rhoSn_extrap = rhofg(Sn, a, T, E1, E0, extr_model, b1, b2)
    eta = current_rho/rhoSn_extrap
    
    Lm = int((L1 + L2)/2)
    Hm = int((H1 + H2)/2)
    c1 = (rhopaw[Lm-1, 1] + rhopaw[Lm, 1] + rhopaw[Lm+1, 1]) / 3.
    c2 = (rhopaw[Hm-1, 1] + rhopaw[Hm, 1] + rhopaw[Hm+1, 1]) / 3.
    rhoL = (nld_lvl[Lm-1, 1] + nld_lvl[Lm, 1] + nld_lvl[Lm+1, 1]) / 3.
    rhoL = max(rhoL, 0.01)
    e1 = a0 + a1 * Lm
    e2 = a0 + a1 * Hm
    _, rhox = rhofg(e2, a, T, E1, E0, extr_model, b1, b2)
    alpha = (np.log(rhoL) + np.log(c2) - np.log(eta*rhox) - np.log(c1)) / (e1-e2)
    Anorm = np.exp(-alpha*e1) * (rhoL)/c1
    c1 = c1/_corrL(a0, a1, nld_lvl, rhopaw, L1, L2, alpha, Anorm)
    c2 = c2/_corrH(a0, a1, rhopaw, H1, H2, alpha, Anorm, a, T, E1, E0, extr_model, b1, b2, eta)
    
    _, rhox = rhofg(e2, a, T, E1, E0, extr_model, b1, b2)
    alpha = (np.log(rhoL) + np.log(c2) - np.log(eta*rhox) - np.log(c1)) / (e1-e2)
    Anorm = np.exp(-alpha*e1) * (rhoL)/c1
    
    #write extrapolation
    
    energies = np.array([a0 + a1*i for i in range(FGmax)])
    extr_mat = np.zeros((len(energies), 2))
    extr_mat[:,0] = energies
    extr_mat[:,1] = np.array([rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)[1]*eta for Ex in energies])
    #extr_mat = np.c_[energies, [rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)[1]*eta for Ex in energies]]
    
    #write rhotmopaw
    rhotmopaw = np.zeros_like(energies)
    for i, Ex in enumerate(energies):
        if i<H1:
            rhotmopaw[i] = Anorm*np.exp(alpha * Ex)*rhopaw[i,1]
        else:
            _, rhox = rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)
            rhotmopaw[i] = eta*rhox
    
    #write normalized rho
    rho_mat = np.zeros((dim, 3))
    for i in range(dim):
        Ex = a0 + a1*i
        rho_mat[i,0] = Ex
        rho_mat[i,1] = Anorm * np.exp(alpha*Ex)*rhopaw[i,1]
        rho_mat[i,2] = Anorm * np.exp(alpha*Ex)*rhopaw[i,2]
    
    #write sigpaw
    sig_mat = np.zeros((dim, 3))
    for i in range(dim):
        Eg = a0 + a1*i
        sig_mat[i,0] = Eg
        sig_mat[i,1] = Anorm * np.exp(alpha*Eg)*sigpaw[i,1]
        sig_mat[i,2] = Anorm * np.exp(alpha*Eg)*sigpaw[i,2]
    
    #extend sig_mat
    sigdimBn = (Sn + 0.5 - a0)/a1 + 0.5
    sigdimBn = int(max(sigdimBn, dim))
    nsigL = _extendL(a0, a1, TL1, TL2, sigdimBn, sig_mat)
    nsigH = _extendH(a0, a1, TH1, TH2, FGmax, sig_mat)
    sigpawext = np.zeros(int(FGmax/5))
    for i in range(int(FGmax/5)):
        if i < TL2 + 1:
            sigpawext[i] = nsigL[i,1]
        elif i < TH1:
            sigpawext[i] = sig_mat[i,1]
        else:
            sigpawext[i] = nsigH[i,1]
    
    #write spincut
    spincut = np.zeros(dim*3)
    sigma2 = np.zeros(FGmax)
    for i in range(max(3*dim, FGmax)):
        Ex = a0 + a1*i
        sig2, _ = rhofg(Ex, a, T, E1, E0, extr_model, b1, b2)
        if i < 3*dim:
            spincut[i] = np.sqrt(sig2)
        if i < FGmax:
            sigma2[i] = sig2
            
    return rho_mat, Anorm, alpha, current_rho, current_drho, extr_mat, sig_mat, rhotmopaw, sigpawext, spincut, sigma2, b1, b2, extr_model, eta

@njit
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

@njit
def _low_levels_chi2(a0, a1, L1, L2, x, y, yerr, rholev):
    '''
    Calculate the chi2 score from the NLD fit to the level density calculated
    from the known levels
    '''
    exp = y[((x >= (a0 + a1*L1)) & (x <= (a0 + a1*L2)))]
    err = yerr[(x >= (a0 + a1*L1)) & (x <= (a0 + a1*L2))]
    theo = rholev[(rholev[:,0] >= (a0 + a1*L1)) & (rholev[:,0] <= (a0 + a1*L2))][:,1]
    chi2_score = chisquared(theo, exp, err, DoF=1, method = 'linear', reduced=False)
    return chi2_score

@njit
def _normalization(a0, a1, Int, D0, current_Gg, sig_mat):
    Fac = Int * D0 / current_Gg
    
    #write gsf
    gsf_rawmat = np.zeros_like(sig_mat)
    for i in range(len(gsf_rawmat)):
        Eg = a0 + a1*i
        gsf_rawmat[i,0] = Eg
        if Eg > 0:
            if Fac == 0.0:
                gsf_rawmat[i,1] = np.nan
                gsf_rawmat[i,2] = np.nan
            else:
                gsf_rawmat[i,1] = sig_mat[i,1]/(Fac*Eg**3)
                gsf_rawmat[i,2] = sig_mat[i,2]/(Fac*Eg**3)
    return gsf_rawmat, Fac
