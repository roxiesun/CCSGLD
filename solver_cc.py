#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 14:32:41 2023

@author: roxiesun
"""

import sys

import autograd.numpy as np
from autograd import grad
from autograd.numpy import sqrt, sin, cos, exp, pi, prod, log
from autograd.numpy.random import normal, uniform, laplace

from time import time
from math import dist


from scipy.stats import multivariate_normal, norm
import scipy.integrate as integrate

class Sampler:
    def __init__(self, f=None, dim=None, boundary=None, xinit=None, partition=None, samplesize=10, rho=20, kappa=1000, gamma=1., parts1=100, parts2=245, lr=0.1, T=1.0, zeta=1, decay_lr=100.):
        
        self.f = f  # the energy function U(x) of interest
        
        self.dim = dim #dimension of the target parameter x
        self.boundary = boundary # NOTE: boundary of the original multi-dimensional x
        self.xinit = np.array(xinit)
        self.partition = partition # for the csgld/ccsgld methods, lower and upper bound of the energy grid
        
        
        self.samplesize = samplesize # batch size for the ccsgld methods ?
        self.rho = rho
        self.kappa =  kappa
        self.gamma = gamma
        self.parts_c = parts1 # the m subregions of csgld
        self.parts_cc = parts2 # the L grids of ccsgld
        self.lr = lr # learning rates epsilon
        self.T = T  # temperature
        self.zeta = zeta # hyperparameter for geometric property of the weight  function Psi
        self.decay_lr = decay_lr # 
    
        
        # initialization for csgld
        self.csgld_beta = self.xinit # note that now beta can be multidimensional
        self.csgld_Gcum = np.array(range(self.parts_c, 0, -1)) * 1.0 / sum(range(self.parts_c, 0, -1))
        self.csgld_div_f = (self.partition[1] - self.partition[0]) / self.parts_c
        self.csgld_J = self.parts_c - 1
        self.csgld_bouncy_move = 0
        self.csgld_grad_mul = 1.
        
        # initialization for ccsgld,
        self.ccsgld_beta = self.xinit # note that now beta can be multidimensional, previously we let beta be the 1d-energy function
        self.ccsgld_div_f = (self.partition[1] - self.partition[0])/(self.parts_cc-1)
        self.ccsgld_logG = np.log(np.ones(self.parts_cc) / self.parts_cc)
        self.ccsgld_zetau = np.ones(self.parts_cc) / self.parts_cc #* 1 / self.ccsgld_div_f 
         # it's like z_1 = partition[0], z_J = partition[1], the left-most and right-most grids
        #self.ccsgld_rawsample = self.xinit
        #self.ccsgld_store_rawsample = self.xinit + np.random.multivariate_normal(np.zeros(self.dim), (4 ** 3) * np.eye(self.dim), size = self.samplesize)#np.empty([self.samplesize, self.dim0])
        #self.accept_count = np.zeros(self.samplesize)
        self.ccsgld_grad_mul = 1.
        self.ccsgld_Jidx = self.parts_cc - 1 # 1 - 244?
        self.ccsgld_sesamples = np.ones(self.samplesize).reshape(self.samplesize, 1)
        self.ccsgld_sigma2 = 1.
        self.ccsgld_store_weight = 0.
        self.ccsgld_bouncy_move = 0
        self.bandwidth = 1.
        
        
    # define for a single sample of 1d-energy 
        
    def in_domain1dU(self, u): return (not (u < self.partition[0] or u > self.partition[1]))
    
    def in_domainU(self, u): return sum(map(lambda i: u[i] < self.partition[0] or u[i] > self.partition[1], range(self.samplesize))) == 0
    
    
    # define for general multi-d-sample
    
    def in_domain(self, beta): return sum(map(lambda i: beta[i] < self.boundary[0] or beta[i] > self.boundary[1], range(self.dim))) == 0

    # note, it returns an interval index 1 ~ 244 if ccsgld_parts = 245
    def find_idx_cc(self, beta): return(min(max(int((np.mean(self.stochastic_f_n(beta)) - self.partition[0]) / self.ccsgld_div_f + 1), 1), self.parts_cc - 1))
    
    def find_idx_u(self, u): return(min(max(int((u - self.partition[0]) / self.ccsgld_div_f + 1), 1), self.parts_cc - 1))
   
   
    def stochastic_f(self, beta): return self.f(beta.tolist()) + laplace(scale = 0.32 * 2**(-1/2))#0.32*normal(size=1) # return array shape (1,)
    
    def stochastic_grad(self, beta): 
        return grad(self.f)(beta) + laplace(scale = 0.32 * 2**(-1/2))#0.32*normal(size=self.dim) # return array shape (dim,)
   
    # evaluating density at an arbitrary point using linear interpolation
    def intpol(self, y):
        idx = self.find_idx_u(y)
        u = (y - (self.partition[0] + (idx - 1)*self.ccsgld_div_f)) / self.ccsgld_div_f
        return (1-u) * exp(self.ccsgld_logG[idx-1]) + u * exp(self.ccsgld_logG[idx])
    
    def logintpol(self, y):
        idx = self.find_idx_u(y)
        u = (y - (self.partition[0] + (idx - 1)*self.ccsgld_div_f)) / self.ccsgld_div_f
        return exp(self.ccsgld_logG[idx]) * exp(u * (self.ccsgld_logG[idx] - self.ccsgld_logG[idx - 1]))
    
    
    # def lattice_refine(self, new_parts):
    #     self.parts = new_parts
    #     self.ccmc_logG = np.zeros(self.parts)
    #     self.ccmc_zeta = np.ones(self.parts) / self.parts
    #     self.div_f = (self.boundary[1] - self.boundary[0])/(self.parts-1)
    
    def find_idx(self, beta): return(min(max(int((self.stochastic_f(beta) - self.partition[0]) / self.csgld_div_f + 1), 1), self.parts_c - 1))
    
    
    def csgld_step(self, iters):
        self.csgld_grad_mul = 1 + self.zeta * self.T * (np.log(self.csgld_Gcum[self.csgld_J]) - np.log(self.csgld_Gcum[self.csgld_J - 1])) / self.csgld_div_f
        proposal = self.csgld_beta - self.lr * self.csgld_grad_mul * self.stochastic_grad(self.csgld_beta) + sqrt(2. * self.lr * self.T) * normal(size=self.dim)
        if self.in_domain(proposal):
            self.csgld_beta = proposal
            
        self.csgld_J = self.find_idx(self.csgld_beta)
        
        step_size = min(self.decay_lr, 10./(iters**0.8+100))
        
        #update theta par via stochastic approximation
        self.csgld_Gcum[:self.csgld_J] = self.csgld_Gcum[:self.csgld_J] + step_size * self.csgld_Gcum[self.csgld_J]**self.zeta * (-self.csgld_Gcum[:self.csgld_J])
        self.csgld_Gcum[self.csgld_J] = self.csgld_Gcum[self.csgld_J] + step_size * self.csgld_Gcum[self.csgld_J]**self.zeta * (1 - self.csgld_Gcum[self.csgld_J])
        self.csgld_Gcum[(self.csgld_J + 1):] = self.csgld_Gcum[(self.csgld_J + 1):] + step_size * self.csgld_Gcum[self.csgld_J]**self.zeta * (-self.csgld_Gcum[(self.csgld_J + 1):])
        
        if self.csgld_grad_mul < 0:
            self.csgld_bouncy_move = self.csgld_bouncy_move + 1
            
  
        
  
    def stochastic_f_n(self, beta): 
        #tmp = map(lambda i: 3.2*normal(size = 1), range(self.samplesize))
        tmp = map(lambda i: laplace(scale = 0.32 * 2**(-1/2)), range(self.samplesize))
        return self.f(beta.tolist()) + np.array(list(tmp)) # return array shape (samplesize, 1)
    
    def stochastic_grad_n(self, beta): 
        #tmp = map(lambda i: 3.2*normal(size = self.dim), range(self.samplesize))
        tmp = map(lambda i: laplace(scale = 0.32 * 2**(-1/2), size = self.dim), range(self.samplesize))
        return grad(self.f)(beta) + np.array(list(tmp)) # return array shape (samplesize, dim)
    
    def deconKernel2(self, u): # note that by using quad_vec, the input u can be a vec
        return integrate.quad_vec(lambda x: (1 / pi) * cos(u * x) * ((1 - x**2)**3) * exp(self.ccsgld_sigma2 * x**2 / (2 * self.bandwidth)), 0. , 1.)
    
    def ccsgld_step(self, iters):
        #self.ccsgld_sesamples = self.stochastic_f_n(self.ccsgld_beta)    
        #self.ccsgld_Jidx = self.find_idx_u(np.mean(self.ccsgld_sesamples))
        
        self.ccsgld_grad_mul = 1 + self.zeta * self.T * (self.ccsgld_logG[self.ccsgld_Jidx] - self.ccsgld_logG[self.ccsgld_Jidx - 1]) / self.ccsgld_div_f
        proposal = self.ccsgld_beta - self.lr * self.ccsgld_grad_mul * np.mean(self.stochastic_grad_n(self.ccsgld_beta), 0) + sqrt(2. * self.lr * self.T) * normal(size=self.dim)
        if self.in_domain(proposal):
            self.ccsgld_beta = proposal
            self.ccsgld_store_weight = exp(self.ccsgld_logG[self.ccsgld_Jidx]) * exp(((np.mean(self.ccsgld_sesamples) - (self.partition[0] + (self.ccsgld_Jidx - 1)*self.ccsgld_div_f)) / self.ccsgld_div_f) * (self.ccsgld_logG[self.ccsgld_Jidx] - self.ccsgld_logG[self.ccsgld_Jidx - 1]))
        
        
        self.ccsgld_sesamples = self.stochastic_f_n(self.ccsgld_beta)
        while not self.in_domainU(self.ccsgld_sesamples):
            outlier = (self.ccsgld_sesamples < self.partition[0]) | (self.ccsgld_sesamples > self.partition[1])
            self.ccsgld_sesamples[(self.ccsgld_sesamples < self.partition[0]) | (self.ccsgld_sesamples > self.partition[1])] = (self.f(self.ccsgld_beta.tolist()) + np.array(list(map(lambda i: 3.2*normal(size = 1), range(int(sum(outlier))))))).reshape(int(sum(outlier)),)
            
        self.ccsgld_Jidx = self.find_idx_u(np.mean(self.ccsgld_sesamples))
        
        #self.ccsgld_sesamples = self.stochastic_f_n(self.ccsgld_beta)    
        #self.ccsgld_Jidx = self.find_idx_u(np.mean(self.ccsgld_sesamples)) # note that find_inx_cc returns the interval where the average of the stochastic energy samples falls into
        
        # Error variance estimation
        step_size_sigma = min(self.decay_lr, 10./(iters**0.6+100))
        #self.ccsgld_sigma2 = np.square(1.2) # assign the true value for now, otherwise
        sig2hat = np.var(self.ccsgld_sesamples, ddof = 1)
        self.ccsgld_sigma2 = (1 - step_size_sigma) * self.ccsgld_sigma2 + step_size_sigma * sig2hat * (1 - self.samplesize / 1e6)
        
        # Density estimate updating 
        # way 1, deconvolution kernel density estimation
        delta = self.rho * self.kappa / max(self.kappa, iters)
        #self.bandwidth = min(delta  ** self.gamma, (self.partition[1] - self.partition[0]) / (2 * (1 + np.log2(self.samplesize)))) ** 2       
        #self.bandwidth = min(delta  ** self.gamma, (max(self.ccsgld_sesamples) - min(self.ccsgld_sesamples)) / (2 * (1 + np.log2(self.samplesize)))) ** 2       
        #self.bandwidth = min(delta  ** self.gamma, (self.ccsgld_sesamples[self.samplesize - 1] - self.ccsgld_sesamples[0]) / (2 * (1 + np.log2(self.samplesize)))) ** 2       
        self.bandwidth = min(delta  ** self.gamma, np.sqrt(2 * self.ccsgld_sigma2 * np.log(self.samplesize))) ** 2
        
        # radius = 4*(sqrt(self.bandwidth))
        
        # sep = int(radius/self.ccsgld_div_f + 1)
        # sep_sample_l = np.array(list(map(lambda k: max(1, self.find_idx_u(self.ccsgld_sesamples[k]) - sep), range(self.samplesize))))
        # sep_sample_u = np.array(list(map(lambda k: min(self.parts_cc - 1, self.find_idx_u(self.ccsgld_sesamples[k]) + sep), range(self.samplesize))))
        # sep_sample = np.column_stack((sep_sample_l, sep_sample_u))
        
        # #evaluating density at the grid points
        
        # for i in range(self.samplesize):
        #     self.ccsgld_zetau[range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)] = 1e-10
        
        # for i in range(self.samplesize):
        #     # if standard gaussian kernel (may suffer from oversmooth error problem)
        #     self.ccsgld_zetau[range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)] += 1 / self.samplesize * (self.bandwidth ** (-1/2)) * 1 / pi * ( 1 / (1 + ((self.partition[0] + np.array(range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2) - 0.5 * self.ccsgld_sigma2 / self.bandwidth * (8 * ((self.partition[0] + np.array(range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2 / (1 + ((self.partition[0] + np.array(range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2)**3) + 0.5 * self.ccsgld_sigma2 / self.bandwidth * (2 / (1 + ((self.partition[0] + np.array(range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2)**2))
        #     # otherwise, a second order kernel with compact and symmetric support
           
        # self.ccsgld_zetau = 1 / self.samplesize * (self.bandwidth ** (-1/2)) * sum(norm.pdf((self.bandwidth ** (-1/2)) * (self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples), scale = sqrt(3.6 - 3.2 ** 2 / self.bandwidth)), 0)
        # self.ccsgld_zetau = 1 / self.samplesize * (self.bandwidth ** (-1/2)) * np.sum(self.deconKernel2((self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples) * (self.bandwidth ** (-1/2)))[0], 0)
        # self.ccsgld_zetau = 1 / self.samplesize * (self.bandwidth ** (-1/2)) * np.sum(np.array(list(map(lambda i: 1 / np.sqrt(2**pi) * exp(-1/2 * ((self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2) * (1 + self.ccsgld_sigma2 / self.bandwidth * (1 - ((self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2)), range(self.samplesize)))), 0)
        
        self.ccsgld_zetau = 1 / self.samplesize * (self.bandwidth ** (-1/2)) * np.sum(np.array(list(map(lambda i: 1 / pi * ( 1 / (1 + ((self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2) - 0.5 * self.ccsgld_sigma2 / self.bandwidth * (8 * ((self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2 / (1 + ((self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2)**3) + 0.5 * self.ccsgld_sigma2 / self.bandwidth * (2 / (1 + ((self.partition[0] + np.arange(self.parts_cc) * self.ccsgld_div_f - self.ccsgld_sesamples[i]) * (self.bandwidth ** (-1/2)))**2)**2)), range(self.samplesize)))), 0)
        
        self.ccsgld_zetau = np.maximum(self.ccsgld_zetau, 1e-40)    
        #self.ccsgld_zetau = self.ccsgld_zetau / (np.sum(self.ccsgld_zetau) * self.ccsgld_div_f)
        self.ccsgld_zetau = self.ccsgld_zetau / (np.sum(self.ccsgld_zetau))
        
        #update the working estimate logG
        step_size = min(self.decay_lr, 10./(iters**0.8+100))
        #step_size = min(self.decay_lr, 10./(iters**0.8+100))
        self.ccsgld_logG = self.ccsgld_logG + step_size * ( self.ccsgld_zetau - (1 / self.parts_cc))
        #self.ccsgld_logG = self.ccsgld_logG + step_size * (np.log(self.ccsgld_zetau) - self.ccsgld_logG)
        #self.ccsgld_logG = self.ccsgld_logG - np.log(np.sum(exp(self.ccsgld_logG) * self.ccsgld_div_f))
        #self.ccsgld_logG = log((exp(self.ccsgld_logG) + step_size * (self.ccsgld_zetau - exp(self.ccsgld_logG)))/ sum(exp(self.ccsgld_logG) + step_size * (self.ccsgld_zetau - exp(self.ccsgld_logG))))
        #self.ccsgld_logG = self.ccsgld_logG - np.log(np.sum(exp(self.ccsgld_logG) * self.ccsgld_div_f))
        self.ccsgld_logG = self.ccsgld_logG - np.log(np.sum(exp(self.ccsgld_logG)))
        
        
        if self.ccsgld_grad_mul < 0:
            self.ccsgld_bouncy_move = self.ccsgld_bouncy_move + 1
        
        # way2, Fourier cosine expansion
        
        # update the working estimate logG
        
    
    
    
    # def ccmc_step(self, iters):
        
    #     #sampling step
    #     s = 0
    #     while s < self.samplesize:
    #         #proposal = self.ccmc_rawsample + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3))
    #         proposal = self.ccmc_store_rawsample[s] + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3))
    #         if self.in_domain(proposal[1]):
    #             #ratio = exp(log(self.f(proposal)) - log(self.intpol(proposal[1])) - log(self.f(self.ccmc_rawsample)) + log(self.intpol(self.ccmc_beta)))
    #             #ratio = (self.f(proposal) * self.intpol(self.ccmc_beta)) / (self.intpol(proposal[1]) * self.f(self.ccmc_rawsample))
    #             ratio = (self.f(proposal) * self.intpol(self.ccmc_store_beta[s])) / (self.intpol(proposal[1]) * self.f(self.ccmc_store_rawsample[s]))
    #             if min(ratio, 1) > np.random.uniform():
    #                 self.accept_count[s] += 1
    #                 self.ccmc_rawsample = proposal
    #                 self.ccmc_beta = proposal[1]
    #                 self.ccmc_store_rawsample[s] = proposal # sth row vector
    #                 self.ccmc_store_beta[s] = proposal[1]
    #             s += 1
        
    #     # proposal = self.ccmc_store_rawsample + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3), size = self.samplesize)
        
    #     # while sum(map(lambda i: self.in_domain(proposal[i,1]), range(self.samplesize))) < self.samplesize :
    #     #     proposal = self.ccmc_store_rawsample + np.random.multivariate_normal(np.zeros(3), (5 ** 3) * np.eye(3), size = self.samplesize)
    #     # # its now an array of shape self.samplesize x self.dim0 
    #     # #if sum(map(lambda i: self.in_domain(proposal[i,1]), range(self.samplesize))) == self.samplesize :
    #     # ratio = np.array(list(map(lambda i: (self.f(proposal[i,:]) * self.intpol(self.ccmc_store_beta[i])) / (self.intpol(proposal[i,1]) * self.f(self.ccmc_store_rawsample[i])), range(self.samplesize))))
    #     # for s in [i for i, x in enumerate(np.minimum(ratio,1).reshape(self.samplesize) > uniform(size = self.samplesize)) if x]:
    #     #     self.ccmc_store_rawsample[s] = proposal[s]
    #     #     self.ccmc_store_beta[s] = proposal[s,1]
    #     # estimate updating
        
    #     # estimate density of the transformed samples y_1, ... y_M by kernel method
    #     delta = self.rho * self.kappa / max(self.kappa, iters)
        
    #     #bandwidth = np.diag([min(delta ** self.gamma, np.ptp(self.ccmc_store_beta[:,0]) / (2 * (1 + np.log2(self.samplesize)))), min(delta ** self.gamma, np.ptp(self.ccmc_store_beta[:,1]) / (2 * (1 + np.log2(self.samplesize))))])
    #     bandwidth = min(delta ** self.gamma, (self.boundary[1] - self.boundary[0]) / (2 * (1 + np.log2(self.samplesize)))) ** 2       
    #     #bandwidth = min(delta ** self.gamma, np.ptp(self.ccmc_store_beta) / (2 * (1 + np.log2(self.samplesize)))) ** 2       
        
    #     radius = 4*(sqrt(bandwidth))
    #     # if we consider fast computation:
    #     sep = int(radius/self.div_f + 1)
    #     sep_sample_l = np.array(list(map(lambda k: max(1, self.find_idx(self.ccmc_store_beta[k]) - sep), range(self.samplesize))))
    #     sep_sample_u = np.array(list(map(lambda k: min(self.parts - 1, self.find_idx(self.ccmc_store_beta[k]) + sep), range(self.samplesize))))
    #     sep_sample = np.column_stack((sep_sample_l, sep_sample_u))
        
        
    #     for i in range(self.samplesize):
    #         self.ccmc_zeta[range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)] = 0.
        
    #     for i in range(self.samplesize):
    #         self.ccmc_zeta[range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)] += 1/self.samplesize * (bandwidth ** (-1/2)) * norm.pdf((bandwidth ** (-1/2)) * (self.boundary[0] + np.array(range(sep_sample[i,0] - 1, sep_sample[i,1] + 1)) * self.div_f - self.ccmc_store_beta[i]))
        
    #     #fast computation: evaluate the kernel density at the grid points lying in max(4ht1, 4ht2) of each sample y_k
    #     #for i in range(self.parts):
    #         #if sum(map(lambda k: abs(self.boundary[0] + i * self.div_f - self.ccmc_store_beta[k]) <= radius, range(self.samplesize))) > 0 :
    #         #self.ccmc_zeta[i] = np.mean(np.array(list(map(lambda k: bandwidth ** (-1/2) * norm.pdf(bandwidth ** (-1/2) * (self.boundary[0] + i * self.div_f - self.ccmc_store_beta[k])), range(self.samplesize)))))
        
    #     # without fast computation, uncomment the line below            
    #     #self.ccmc_zeta = np.mean(np.array(list(map(lambda k: (bandwidth ** (-1/2)) * norm.pdf((bandwidth ** (-1/2)) * (self.boundary[0] + np.array(range(self.parts)) * self.div_f - self.ccmc_store_beta[k])), range(self.samplesize)))), axis = 0)
    #     #normalizing
    #     self.ccmc_zeta = self.ccmc_zeta / np.sum(self.ccmc_zeta)
        
    #     #update the working estimate logG
    #     self.ccmc_logG = self.ccmc_logG + delta * (self.ccmc_zeta - (1 / self.parts))
    #     #if add a constrain s.t. \sum_\xi_{zij} = self.parts/61?
    #     #self.ccmc_logG = log(exp(self.ccmc_logG)/sum(exp(self.ccmc_logG))*1/self.div_f)
        
        
        
    #     #check if we need to refine the lattice
    #     if self.div_f / sqrt(bandwidth) > 8:
    #         print("Bandwidth too small: d/h is %f \n" % (self.div_f / sqrt(bandwidth)))
    #         sys.exit('Lattice refining required: ')
    #         #self.lattice_refine(self.parts*2)
                
                  
        
        
        