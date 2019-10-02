#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:47:10 2019

@author: nvthaomy
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import simps
import spline, sys, argparse, re
import matplotlib.pyplot as plt
from math import ceil, log, floor
import os


"""Fitting Gaussians to spline using least squares 
   obj: objective function, calculates the Boltzmann weighted residuals 
   constrains: 
       Gaussians with even index to be repulsive, and odd index to be attractive
       upper bound of repulsive Gaussian is maximum potential from spline, can modify in getBounds
   by default, optimize with incremental number of Gaussians. Initial guess for 1st Gaussian opt is B = max value of spline potential, K = 1,
       initial values for the remaining opt are optimized parameters from previous opt + [B=0,K=0] for the newly added Gaussian
   can also optimize in one stage by providind intial values and using -nostage flag

   Outputs:
       Gaussian parameters,energy scale and kappa, in the form: [B1, K1, B2, K2,...,Bn,Kn]   
   """
#test command
#python spline2gaussians-leastsquares.py  -k "2.7835e+02 , 3.3541e+00 , -5.8015e-01, 1.6469e-01 ,-1.1965e-01, 5.2720e-02 , -2.3451e-02, 2.6243e-03" -cut 11 -n 2

def GaussianBasisLSQ(knots, rcut, rcutinner, ng, nostage, N, BoundSetting, U_max_2_consider, 
                        SlopeCut=-10., ShowFigures=False, SaveToFile=True, SaveFileName = 'GaussianLSQFitting',
                        weight_rssq = False, Cut_Length_Scale=1., TailCorrection=False, TailWeight=1E6):
    
    n = ng
    knots = [float(i) for i in re.split(' |,',knots) if len(i)>0]
    
    SlopeCut = SlopeCut # Setting for outpicking the number of Gaussians to use in basis set
    ShowFigures = ShowFigures
    SaveToFile = SaveToFile
    SaveFileName = SaveFileName

    if SaveToFile: 
        try:
            os.mkdir(SaveFileName)
        except:
            pass 
        os.chdir(SaveFileName)

    def ceil_power_of_10(n):
        exp = log(n, 10)
        exp = ceil(exp)
        return 10**exp
        
    def floor_power_of_10(n):
        exp = log(n, 10)
        exp = floor(exp)
        return 10**exp

    def obj(x,w,rs,u_spline,TailCorrection,rcut,TailWeight): 
        """Calculate Boltzmann weighted residuals"""
        n = int(len(x)/2) #number of Gaussians
        u_gauss = getUgauss(x,rs,n)
        if TailCorrection:
            w_tail = TailWeight
            rcut_temp = np.zeros(1)
            rcut_temp[0] = rcut
            tail_value = np.abs(getUgauss(x,rcut_temp,n)-0)
            return w*(u_gauss-u_spline)+w_tail*(tail_value)
        else:
            return w*(u_gauss-u_spline)

    def getUgauss(x,rs,n):
        u_gauss = np.zeros(len(rs))
        for i in range(n):
            B = x[i*2]
            K = x[i*2+1]
            u_gauss += B*np.exp(-K*rs**2)
        return u_gauss

    def getUspline(knots, rcut, rs, MaxPairEnekBT = 20, kB = 1,TempSet = 1):
        """calculate spline potential and adjust the hard core region
        to match the tabulated pair potentials generated by sim 
        (PotentialTablePair class in sim/export/lammps.py)"""
        myspline = spline.Spline(rcut,knots)
        u_spline = []
        du_spline = []
        for r in rs:
            u_spline.append(myspline.Val(r))
            du_spline.append(myspline.DVal(r)) 
        u_spline = np.array(u_spline)
        #get the maximum pair energy
        MaxPairEne = kB * TempSet * MaxPairEnekBT
        #indices where energy is greater
        ind = np.where(u_spline> MaxPairEne)[0]
        if len(ind):
            #find the first index where energy is valid
            i = ind[-1] + 1
            #do a linear extrapolation in the hard core region
            u_spline[:i] = (rs[i] - rs[:i]) * -du_spline[i] + u_spline[i]
            for j in ind:
                du_spline[j] = du_spline[i]
        return u_spline,du_spline
        
    def weight(rs,u_spline, weight_rssq):
        rssq = np.multiply(rs,rs)
        w = np.exp(-u_spline)
        if weight_rssq: w = np.multiply(w,rssq)
        w = w#/np.sum(w)
        return w

    autofindrcutinner = False
    if U_max_2_consider != None: 
        autofindrcutinner = True
        rcutinner = 0.
        
    rs = np.linspace(rcutinner,rcut,N)
    u_spline, du_spline = getUspline(knots,rcut,rs)
    u_max = np.max(u_spline)
    np.savetxt('u_spline.data', u_spline)
    
    # Get the integral of the pair potential and the second virial coeff
    rs_temp = np.linspace(0,rcut,1E6)
    rssq = np.multiply(rs_temp,rs_temp)
    u_spline_temp, du_spline_temp = getUspline(knots,rcut,rs_temp)
    integrand_pot = np.multiply(u_spline_temp,rssq)
    integrand_virial = 4*np.pi*np.multiply(rssq,(1-np.exp(-u_spline_temp)))
    int_pot = simps(integrand_pot,rs_temp)
    int_virial = simps(integrand_virial,rs_temp)
    
    if autofindrcutinner:
        for i,val in enumerate(u_spline):
            if val < U_max_2_consider and i == 0:
                rcutinner = rs[i]
            elif val < U_max_2_consider and u_spline[i-1] > U_max_2_consider:
                rcutinner = rs[i]
                
        rs = np.linspace(rcutinner,rcut,N)
        u_spline, du_spline = getUspline(knots,rcut,rs)
        u_max = np.max(u_spline)   
        

    w = weight(rs,u_spline, weight_rssq)
    np.savetxt('weights.data',zip(rs,w))
    kappa_lowerbound = Cut_Length_Scale/rcut**2

    def getBounds(n, BoundSetting):
        bounds = ([],[]) 
        lower_energybound = -np.inf
        upper_energybound = u_max
        if BoundSetting == 'Option1':
            for i in range(n):
                if i % 2 == 0: #bounds of repulsive Gaussian
                    bounds[0].extend([0,kappa_lowerbound]) #lower bound of B and K
                    bounds[1].extend([upper_energybound,np.inf]) #upper bound of B and K
                else: #bounds of attractive Gaussian
                    bounds[0].extend([lower_energybound,kappa_lowerbound])
                    bounds[1].extend([0,np.inf])
        elif BoundSetting == 'Option2': # All float positive and negative
            for i in range(n):  
                bounds[0].extend([lower_energybound,0]) #lower bound of B and K
                bounds[1].extend([upper_energybound,np.inf]) #upper bound of B and K
                
        return bounds
        
    def plot(xopt,rs,n,u_spline, ShowFigures):
        u_gauss = getUgauss(xopt,rs,n)
        plt.figure()
        plt.plot(rs,u_spline,label="spline",linewidth = 3)
        plt.plot(rs,u_gauss,label="{}-Gaussian".format(n),linewidth = 3)
        plt.scatter(np.linspace(0,rcut,len(knots)),knots,label = "spline knots",c='r')
        plt.ylim(min(np.min(u_spline),np.min(u_gauss))*2)
        plt.xlim(0,rcut)
        plt.xlabel('r')
        plt.ylabel('u(r)')
        plt.legend(loc='best')
        plt.savefig('NumberGauss_{}.pdf'.format(n))
        if ShowFigures:
            plt.show()
        plt.close()

    cost_list = []
    gauss_list = []
    param_list = []
    logout = open("fitting.data",'w')
    logout.close()
    u_gauss_list = []                 
 
    if nostage == False:   
        for i in range(n):
            logout = open("fitting.data",'a')
            bounds = getBounds(i+1,BoundSetting)
            if i == 0:
                x0 = np.array([u_max,1.]) #initial vals for B and kappa of first Gaussian
                sys.stdout.write('\nInitial guess for 1st Gaussian:')
                sys.stdout.write('\nB: {}, K: {}'.format(x0[0],x0[1]))
                sys.stdout.write('\nParameters from optimizing {} Gaussian:'.format(i+1))            

            else:
                x0 = [p for p in xopt]
                x0.extend([0,kappa_lowerbound])
                sys.stdout.write('\nInitial guess: {}'.format(x0))
                sys.stdout.write('\nParameters from optimizing {} Gaussians:'.format(i+1))
                logout.write('\nParameters from optimizing {} Gaussians:\n'.format(i+1))
                logout.write('\nInitial guess: {}\n'.format(x0))
                
            gauss = least_squares(obj,x0, args = (w,rs,u_spline, TailCorrection, rcut,TailWeight),bounds=bounds)
            xopt = gauss.x
            param_list.append(xopt)
            sys.stdout.write('\n{}'.format(xopt))
            sys.stdout.write('\nLSQ: {}\n'.format(gauss.cost))
            logout.write('\n{}'.format(xopt))
            logout.write('\nLSQ: {}\n'.format(gauss.cost))
            logout.close()
            plot(xopt,rs,i+1,u_spline, ShowFigures)
            u_gauss = getUgauss(xopt,rs,i+1)
            u_gauss_list.append(u_gauss)                                                                                 
            cost_list.append(gauss.cost)
            gauss_list.append(i+1)

    else:
        if len(args.x0) == 0:
            raise Exception('Need initial values of Gaussian parameters')
        else:
            x0 = [float(i) for i in re.split(' |,',args.x0) if len(i)>0]
            if len(x0) != 2*n:
                raise Exception('Wrong number of initial values')
        bounds = getBounds(n)
        sys.stdout.write('\nInitial guess:')
        sys.stdout.write('\n{}'.format(x0))
        gauss = least_squares(obj,x0, args = (w,rs,u_spline, TailCorrection, rcut,TailWeight),bounds=bounds)
        xopt = gauss.x
        sys.stdout.write('\nParameters from optimizing {} Gaussians:'.format(n))
        sys.stdout.write('\n{}'.format(xopt))
        sys.stdout.write('\nLSQ: {}\n'.format(gauss.cost))
        plot(xopt,rs,n,u_spline, ShowFigures)
        u_gauss = getUgauss(xopt,rs,i+1)
        u_gauss_list.append(u_gauss)                                                                                 
        cost_list.append(gauss.cost)
        gauss_list.append(i+1)

    ''' One method to pick optimal number of Gaussians. '''
    derLSQObj = []
    index_opt = None
    logout = open("fitting.data",'a')
    for i,val in enumerate(cost_list):        
        if i == 0:
            pass
        else:
            der_temp = val - cost_list[i-1]
            derLSQObj.append(der_temp)
            if der_temp > SlopeCut and index_opt == None:
                index_opt = i-1
            elif index_opt != None and cost_list[i-1]/val > 2.:                
                index_opt = i
            elif i == (len(cost_list)-1) and index_opt == None:
                index_opt = i # Pick the last one so nothing crashes or if only one is specified
            logout.write('derivative: {}\n'.format(der_temp))
            logout.write('Val[i-1]/val: {}\n'.format(cost_list[i-1]/val))
    
    Override = False
    if Override:
        index_opt = 2
    
    logout.write('\n The integral of the pair potential:\n')
    logout.write('{}'.format(int_pot))
    logout.write('\n The second virial coefficient:\n')
    logout.write('{}'.format(int_virial))
    
    logout.write('\nOptimal number of Gaussians are {}\n'.format(gauss_list[index_opt]))
    logout.write('Optimal parameters: \n')
    logout.write('\n{}'.format(param_list[index_opt]))
    logout.write('\nLSQ: {}\n'.format(cost_list[index_opt]))
    logout.write('TailCorrection: {}'.format(TailCorrection))
    logout.write('\nTail Weight: {}'.format(TailWeight))
    logout.write('\nTailCorrection constrains the Gaussians to sum to zero at the cutoff.')
    logout.close()        
    np.savetxt('slope.data', zip(gauss_list[1:],derLSQObj))        

    u_gauss = getUgauss(xopt,rs,n)
    plt.figure()
    plt.plot(rs,u_spline,label="spline",linewidth = 3)
    plt.plot(rs,u_gauss,label="{}-Gaussian".format(n),linewidth = 3)
    plt.scatter(np.linspace(0,rcut,len(knots)),knots,label = "spline knots",c='r')
    plt.ylim(min(np.min(u_spline),np.min(u_gauss))*1.25,2)
    plt.xlim(0,rcut)
    plt.xlabel('r')
    plt.ylabel('u(r)')
    plt.legend(loc='best')
    plt.savefig('GaussFit.pdf')
    if ShowFigures:
        plt.show()

    np.savetxt('cost.data', zip(gauss_list,cost_list))
    np.savetxt('u_gauss.data', np.transpose(np.asarray(u_gauss_list)))                                                                  

    plt.figure()
    plt.semilogy(gauss_list,cost_list,label='cost',marker='o', markersize=6)
    plt.ylim(floor_power_of_10(np.min(np.asarray(cost_list))),ceil_power_of_10(np.max(np.asarray(cost_list))))
    plt.xlabel('number Gaussians')
    plt.ylabel('LSQ Objective')
    plt.savefig('LSQ.pdf')
    if ShowFigures:
        plt.show()

    if SaveToFile: os.chdir('..')
    
    return [gauss_list[index_opt], param_list[index_opt], u_gauss_list, u_spline, rs]
