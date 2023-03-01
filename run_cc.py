#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 15:12:51 2023

@author: roxiesun
"""

from solver_cc import Sampler
import argparse
import time

import autograd.numpy as np
from autograd import grad
from autograd.numpy import log, sqrt, sin, cos, exp, pi, prod
from autograd.numpy.random import normal, uniform

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal, norm
from scipy.integrate import cumtrapz, quad
from scipy.signal import savgol_filter
from sklearn.neighbors import KernelDensity


parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-samplesize', default=100, type=int, help='batchsize')
parser.add_argument('-rho', default=20, type=int, help='For gain factor')
parser.add_argument('-kappa', default=2000, type=int, help='For gain factor')
parser.add_argument('-gamma', default=1., type=float, help='For bandwidth')
parser.add_argument('-parts1', default=244, type=int, help='Total numer of partitions for csgld')
parser.add_argument('-parts2', default=245, type=int, help='Total numer of grid points for ccsgld')
parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('-T', default=1, type=float, help='inverse temperature')
parser.add_argument('-zeta', default=0.75, type=float, help='Adaptive hyperparameter')
parser.add_argument('-decay_lr', default=3e-3, type=float, help='Decay lr')
parser.add_argument('-seed', default=1, type=int, help='seed')
parser.add_argument('-flag', default=1, type=int, help='Job Array Index')
parser.add_argument('-nrep', default=1, type=int, help='Number of replications')
pars = parser.parse_args()

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
#np.random.seed(pars.seed)
flag = pars.flag

ccsgld_bias = []
csgld_bias = []
ccsgld_rmse = []
csgld_rmse = []

split_, total_ = 20, 8e5

st = time.time()

def mixture(x):
    return ((x[0]**2 + x[1]**2)/10 - (cos(1.2*pi*x[0]) + cos(1.2*pi*x[1]))) / 0.3 + ((x[0]**2 + x[1]**2) > 7) * ((x[0]**2 + x[1]**2) - 7)
        

def mixture_expand(x, y): return mixture([x, y])
def function_plot(x, y): return np.exp(-mixture([x, y]))
#def logfunction_plot(x): return np.log(mixture(x)) # get log p.d.f if mixture(x) is the p.d.f rather than energy function

boundary_ = 2.5
axis_x = np.linspace(-boundary_, boundary_, 500)
axis_y = np.linspace(-boundary_, boundary_, 500)
axis_X, axis_Y = np.meshgrid(axis_x, axis_y)

energy_grid = mixture_expand(axis_X, axis_Y)
prob_grid = function_plot(axis_X, axis_Y)
lower_bound, upper_bound = np.min(energy_grid) - 1, np.max(energy_grid) + 1

for irep in range(int(pars.nrep)):
    np.random.seed(int(pars.seed*1000) + int(irep))
    sampler = Sampler(f=mixture, dim=2, boundary=[-boundary_, boundary_], xinit=[2., 2.],
                      partition=[lower_bound, upper_bound], \
                      samplesize=pars.samplesize, rho=pars.rho, kappa=pars.kappa, gamma=pars.gamma, parts1=pars.parts1, \
                      parts2=pars.parts2, lr=pars.lr, T=pars.T, zeta=pars.zeta, decay_lr=pars.decay_lr)

    # Compute the exact energy p.d.f., note that this gives exactly the true histogram estimate of the energy pdf (with the area of the bars adding up to 1), which is the target of CSGLD
    exact_energy_pdf = []
    energy_unit = (upper_bound - lower_bound) * 1.0 / sampler.parts_c
    exact_energy_grids = lower_bound + np.arange(sampler.parts_c) * energy_unit
    fine_axis_x = np.linspace(-boundary_, boundary_, 500)
    fine_axis_y = np.linspace(-boundary_, boundary_, 500)
    fine_axis_X, fine_axis_Y = np.meshgrid(fine_axis_x, fine_axis_y)

    fine_energy_grid = mixture_expand(fine_axis_X, fine_axis_Y)
    fine_prob_grid = function_plot(fine_axis_X, fine_axis_Y)
    fine_prob_grid /= fine_prob_grid.sum()
    for ii in range(sampler.parts_c):
        tag = (fine_energy_grid > lower_bound + ii * energy_unit) & (
                    fine_energy_grid < lower_bound + (ii + 1) * energy_unit)
        exact_energy_pdf.append(fine_prob_grid[tag].sum())

    # exact energy p.d.f for ccsgld with larger, e.g., 101 or 245 grid points

    exact_energy_pdf_2 = []
    energy_unit_2 = (upper_bound - lower_bound) * 1.0 / (sampler.parts_cc - 1)
    exact_energy_grids_2 = lower_bound + np.arange(sampler.parts_cc) * energy_unit_2

    for ii in range(sampler.parts_cc - 1):
        tag = (fine_energy_grid > lower_bound + ii * energy_unit) & (
                    fine_energy_grid < lower_bound + (ii + 1) * energy_unit)
        exact_energy_pdf.append(fine_prob_grid[tag].sum())

    kde_sk = KernelDensity(bandwidth=0.5, kernel='gaussian')
    kde_sk.fit(fine_energy_grid.reshape([-1, 1]))

    exact_energy_pdf_2 = np.exp(kde_sk.score_samples(exact_energy_grids_2.reshape(-1, 1)))
    # sns.kdeplot(fine_energy_grid.flatten())
    # plt.plot(exact_energy_grids_2, exact_energy_pdf_2)

    warmup = 15000
    csgld_x = np.array([sampler.csgld_beta])  # to store the value of x (denoted by beta here) ACROSS ITERATIONS
    csgld_importance_weights = [0., ]
    jump_col = ['blue', 'red']
    csgld_history_samples = np.zeros(shape=(0, 180))

    ccsgld_x = np.array([sampler.ccsgld_beta])  # to store the value of x (denoted by beta here) ACROSS ITERATIONS
    ccsgld_importance_weights = [0., ]
    ccsgld_history_samples = np.zeros(shape=(0, sampler.parts_cc))
    # ccsgld_density = ??
    # ccmc_density = np.array([sampler.ccmc_zeta]) # Note: bracket here is necessary for later np.vstack, stack along layers/iters

    # ccmc_weights = exp(np.array([sampler.ccmc_logG]))
    # ccmc_store_beta = np.array([sampler.ccmc_store_beta])

    for iters in range(int(total_)):
        sampler.csgld_step(iters)
        sampler.ccsgld_step(iters)
        if iters > warmup:
            if iters % split_ == 0:
                # note: axis 0: columns, axis 1:rows, or
                #      axis 0: , axis 1: cols, axis 2: rows
                csgld_x = np.vstack((csgld_x, sampler.csgld_beta))
                csgld_importance_weights.append(sampler.csgld_Gcum[sampler.csgld_J] ** pars.zeta)

                ccsgld_x = np.vstack((ccsgld_x, sampler.ccsgld_beta))
                ccsgld_importance_weights.append(sampler.ccsgld_store_weight ** pars.zeta)

            if iters % 10000 == 0:  # and len(csgld_x[:,1]) > 5:
                csgld_history_samples = np.vstack((csgld_history_samples, sampler.csgld_Gcum[:180]))
                ccsgld_history_samples = np.vstack((ccsgld_history_samples, exp(sampler.ccsgld_logG)))
                # col_std = np.std(csgld_history_samples, 0) / sqrt(csgld_history_samples.shape[0])
                # fig = plt.figure(figsize=(13, 13))
                fig = plt.figure(figsize=(16, 10))

                plt.subplot(2, 2, 1).set_title('(a) CSGLD (samples from the flattened density)', fontsize=15)
                plt.contour(axis_X, axis_Y, prob_grid)
                plt.yticks([-4, -2, 0, 2, 4])
                plt.plot(csgld_x[:, 0][:-5], csgld_x[:, 1][:-5], linewidth=0.1, marker='.', markersize=2, color='k',
                         label="Iteration=" + str(iters))
                col_type = 0 if sampler.csgld_grad_mul > 0 else 1
                plt.plot(csgld_x[:, 0][-4], csgld_x[:, 1][-4], linewidth=0.15, marker='.', markersize=4,
                         color=jump_col[col_type], alpha=1, label="Bouncy moves=" + str(sampler.csgld_bouncy_move));
                plt.plot(csgld_x[:, 0][-3], csgld_x[:, 1][-3], linewidth=0.2, marker='.', markersize=6,
                         color=jump_col[col_type], alpha=1, label="Grad multiplier=" + str(sampler.csgld_grad_mul)[:4]);
                if sampler.csgld_Gcum[sampler.csgld_J] ** pars.zeta < 1e-4:
                    plt.plot(csgld_x[:, 0][-2], csgld_x[:, 1][-2], linewidth=0.25, marker='.', markersize=8,
                             color=jump_col[col_type], alpha=1, label="Importance weight=0");
                else:
                    plt.plot(csgld_x[:, 0][-1], csgld_x[:, 1][-1], linewidth=0.3, marker='.', markersize=10,
                             color=jump_col[col_type], alpha=1,
                             label="Importance weight=" + str(sampler.csgld_Gcum[sampler.csgld_J] ** pars.zeta)[:4]);
                plt.legend(loc="upper left", prop={'size': 12})

                plt.subplot(2, 2, 2).set_title('(b) CCSGLD (samples from the flattened density)', fontsize=15)
                plt.contour(axis_X, axis_Y, prob_grid)
                plt.yticks([-4, -2, 0, 2, 4])
                plt.plot(ccsgld_x[:, 0][:-5], ccsgld_x[:, 1][:-5], linewidth=0.1, marker='.', markersize=2, color='k',
                         label="Iteration=" + str(iters))
                col_type = 0 if sampler.ccsgld_grad_mul > 0 else 1
                plt.plot(ccsgld_x[:, 0][-4], ccsgld_x[:, 1][-4], linewidth=0.15, marker='.', markersize=4,
                         color=jump_col[col_type], alpha=1, label="Bouncy moves=" + str(sampler.ccsgld_bouncy_move));
                plt.plot(ccsgld_x[:, 0][-3], ccsgld_x[:, 1][-3], linewidth=0.2, marker='.', markersize=6,
                         color=jump_col[col_type], alpha=1,
                         label="Grad multiplier=" + str(sampler.ccsgld_grad_mul)[:4]);
                if sampler.ccsgld_store_weight ** pars.zeta < 1e-4:
                    plt.plot(ccsgld_x[:, 0][-2], ccsgld_x[:, 1][-2], linewidth=0.25, marker='.', markersize=8,
                             color=jump_col[col_type], alpha=1, label="Importance weight=0");
                else:
                    plt.plot(ccsgld_x[:, 0][-1], ccsgld_x[:, 1][-1], linewidth=0.3, marker='.', markersize=10,
                             color=jump_col[col_type], alpha=1,
                             label="Importance weight=" + str(sampler.ccsgld_store_weight ** pars.zeta)[:4]);
                plt.legend(loc="upper left", prop={'size': 12})

                # plt.subplot(2, 2, 3).set_title('(c) Energy log p.d.f estimate', fontsize=14)
                # #plt.cla()
                # plt.plot(log(sampler.csgld_Gcum)[:180], color='red', label='CSGLD_Est.', linewidth=1.5)
                # plt.plot(log(exact_energy_pdf)[:180], color='black', label='Ground truth', linewidth=1.5)
                # plt.legend(loc='upper right', prop={'size': 10})
                # plt.ylim([-20, 0.15])
                plt.subplot(2, 2, 3).set_title('(c) Energy p.d.f estimate', fontsize=15)
                # plt.cla()
                plt.plot(sampler.csgld_Gcum[:180], color='red', label='CSGLD_Est.', linewidth=1.5)
                plt.plot(exact_energy_pdf[:180], color='black', label='Ground truth', linewidth=1.5)
                plt.legend(loc='upper right', prop={'size': 12})
                plt.ylim([0, 0.15])
                # plt.gca().axes.xaxis.set_visible(False)
                # plt.gca().axes.yaxis.set_visible(False)
                plt.annotate("Higher energy", fontsize=12, xy=(155, 0.005), xytext=(125, 0.03),
                             arrowprops=dict(arrowstyle="->"))
                if len(csgld_x[:, 1]) > 5:
                    col_std = np.std(csgld_history_samples, 0) / sqrt(csgld_history_samples.shape[0])
                    plt.fill_between(range(180), sampler.csgld_Gcum[:180] - 10 * col_std,
                                     sampler.csgld_Gcum[:180] + 10 * col_std, color='red', alpha=.3)

                    # plt.fill_between(range(180), log(sampler.csgld_Gcum)[:180]-10*col_std, log(sampler.csgld_Gcum)[:180]+10*col_std, color='red', alpha=.3)

                plt.subplot(2, 2, 4).set_title('(d) Energy p.d.f estimate', fontsize=15)
                # plt.cla()
                # plt.plot(exp(sampler.ccsgld_logG), color='red', label='CCSGLD_Est.', linewidth=1.5)
                # plt.plot(exact_energy_pdf_2, color='black', label='Ground truth', linewidth=1.5)
                # plt.plot(exact_energy_grids_2, sampler.ccsgld_logG, color='red', label='CCSGLD_Est.', linewidth=1.5)
                # plt.plot(exact_energy_grids_2, log(sampler.ccsgld_zetau), color='yellow', label='zetau.', linewidth=1.5)
                # plt.plot(exact_energy_grids_2,log(exact_energy_pdf_2), color='black', label='Ground truth', linewidth=1.5)

                # plt.plot(sampler.ccsgld_logG[:180], color='red', label='CCSGLD_Est.', linewidth=1.5)
                # plt.plot(log(exact_energy_pdf[:180]), color='black', label='Ground truth', linewidth=1.5)
                # plt.legend(loc='upper right', prop={'size': 10})
                # plt.ylim([-20, 0.15])
                plt.plot(exp(sampler.ccsgld_logG)[:180], color='red', label='CCSGLD_Est.', linewidth=1.5)
                plt.plot(exact_energy_pdf[:180], color='black', label='(Discrete) Ground truth', linewidth=1.5)
                plt.legend(loc='upper right', prop={'size': 12})
                plt.ylim([0, 0.15])
                plt.annotate("Higher energy", fontsize=12, xy=(155, 0.005), xytext=(125, 0.03),
                             arrowprops=dict(arrowstyle="->"))

                # plt.gca().axes.xaxis.set_visible(False)
                # plt.gca().axes.yaxis.set_visible(False)
                # plt.annotate("Higher energy", fontsize=10, xy=(85, 0.005), xytext=(65, 0.03), arrowprops=dict(arrowstyle="->"))
                if len(ccsgld_x[:, 1]) > 5:
                    # col_std = np.std(ccsgld_history_samples, 0) / sqrt(ccsgld_history_samples.shape[0])
                    # plt.fill_between(range(sampler.parts_cc), exp(sampler.ccsgld_logG[:80])-10*col_std, exp(sampler.ccsgld_logG[:80])+10*col_std, color='red', alpha=.3)
                    col_std = np.std(ccsgld_history_samples[:, :180], 0) / sqrt(ccsgld_history_samples.shape[0])
                    plt.fill_between(range(180), exp(sampler.ccsgld_logG)[:180] - 10 * col_std,
                                     exp(sampler.ccsgld_logG)[:180] + 10 * col_std, color='red', alpha=.3)

                    # plt.fill_between(range(180), sampler.ccsgld_logG[:180]-10*col_std, sampler.ccsgld_logG[:180]+10*col_std, color='red', alpha=.3)

                # ax = fig.add_subplot(2, 1, 1)
                # ax.set_title('(a) True v.s. estimated log-density, itr = %d' % (iters), fontsize=16)
                # plt.plot(axis_x, logprob_grid, '--', label='True')
                # plt.plot(axis_x, sampler.ccmc_logG, label='Est.')
                # #plt.plot(axis_x, np.median(np.log(ccmc_weights), axis = 0), label='Post.Median.')
                # plt.plot(axis_x, np.mean(np.log(ccmc_weights), axis = 0), label='Post.Mean.')
                # ax.set_xlabel('$\lambda$(x)', fontsize=13)
                # ax.set_ylabel('logPDF', fontsize=13)
                # ax.set_ylim(-25, 5)
                # legend = ax.legend(loc='upper left',prop={'size': 11})

                # ax = fig.add_subplot(2, 1, 2)
                # plt.hist(ccmc_store_beta.flatten(), bins= 'auto')
                # ax.set_title("(b) Histogram of $\lambda$(x) samples, itr = %d" % (iters), fontsize=16)

                plt.tight_layout()
                # on mac
                # plt.savefig('/Users/roxiesun/OneDrive - The Chinese University of Hong Kong/2022/purdue/research/ccmc_try/pics1/' + '{:04}'.format((iters - warmup)//1000) + '.png')
                # on officePC
                plt.savefig(
                    '{:02}'.format(flag) + '_' + '{:03}'.format(irep+1) + '_' + '{:04}'.format(
                        (iters - warmup) // 10000) + '.png')
                # plt.show()
                # plt.close(fig)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # resample the parameters by considering the importance weights
    scaled_importance_weights = csgld_importance_weights / np.mean(csgld_importance_weights)

    resample_x = np.empty((0, 2))
    for i in range(len(csgld_x)):
        while scaled_importance_weights[i] > 1:
            tag = np.random.binomial(1, p=min(1, scaled_importance_weights[i]))
            scaled_importance_weights[i] -= 1
            if tag == 1:
                resample_x = np.vstack((resample_x, csgld_x[i,]))

    split_ = 1

    fig = plt.figure(figsize=(16, 7.15))
    plt.subplot(2, 3, 1).set_title('(a) Ground truth')
    sns.heatmap(prob_grid, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)

    warm_sample = 50
    plt.subplot(2, 3, 2).set_title('(b) CSGLD (before resampling)')
    ax = sns.kdeplot(csgld_x[:, 0][::split_][warm_sample:], csgld_x[:, 1][::split_][warm_sample:], bw_adjust=1,
                     cmap="Blues", shade=True, shade_lowest=False)
    ax.set_xlim(-boundary_, boundary_)
    ax.set_ylim(-boundary_, boundary_)

    plt.subplot(2, 3, 3).set_title('(c) CSGLD (after resampling)')
    ax = sns.kdeplot(resample_x[:, 0][::split_][warm_sample:], resample_x[:, 1][::split_][warm_sample:], bw_adjust=1,
                     cmap="Blues", shade=True, shade_lowest=False)
    ax.set_xlim(-boundary_, boundary_)
    ax.set_ylim(-boundary_, boundary_)

    # fig = ax.get_figure()

    # resampling procedure for CCSGLD
    cc_scaled_importance_weights = ccsgld_importance_weights / np.mean(ccsgld_importance_weights)

    cc_resample_x = np.empty((0, 2))
    for i in range(len(ccsgld_x)):
        while cc_scaled_importance_weights[i] > 1:
            tag = np.random.binomial(1, p=min(1, cc_scaled_importance_weights[i]))
            cc_scaled_importance_weights[i] -= 1
            if tag == 1:
                cc_resample_x = np.vstack((cc_resample_x, ccsgld_x[i,]))

    split_ = 1

    # fig = plt.figure(figsize=(10, 3.15))
    plt.subplot(2, 3, 4).set_title('(a) Ground truth')
    sns.heatmap(prob_grid, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)

    warm_sample = 50
    plt.subplot(2, 3, 5).set_title('(d) CCSGLD (before resampling)')
    ax = sns.kdeplot(ccsgld_x[:, 0][::split_][warm_sample:], ccsgld_x[:, 1][::split_][warm_sample:], bw_adjust=1,
                     cmap="Blues", shade=True, shade_lowest=False)
    ax.set_xlim(-boundary_, boundary_)
    ax.set_ylim(-boundary_, boundary_)

    plt.subplot(2, 3, 6).set_title('(e) CCSGLD (after resampling)')
    ax = sns.kdeplot(cc_resample_x[:, 0][::split_][warm_sample:], cc_resample_x[:, 1][::split_][warm_sample:],
                     bw_adjust=1, cmap="Blues", shade=True, shade_lowest=False)
    ax.set_xlim(-boundary_, boundary_)
    ax.set_ylim(-boundary_, boundary_)

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig('{:02}'.format(flag) + '_' + '{:03}'.format(irep+1) + '_Contour.png')
            
    csgld_bias.append(np.mean(resample_x[:,0]+resample_x[:,1]))
    ccsgld_bias.append(np.mean(cc_resample_x[:,0]+cc_resample_x[:,1]))
    csgld_rmse.append(sqrt(np.mean(resample_x[:,0]+resample_x[:,1])**2 + np.var(resample_x[:,0]+resample_x[:,1])))
    ccsgld_rmse.append(sqrt(np.mean(cc_resample_x[:,0]+cc_resample_x[:,1])**2 + np.var(cc_resample_x[:,0]+cc_resample_x[:,1])))
    print('Replication_', irep + 1, 'of job ', flag,  'done and recorded. \n')

with open('{:02}'.format(flag) + '_' + "_csgld_Bias.txt", "w") as fp:
    fp.write("\n".join(str(item) for item in csgld_bias))

with open('{:02}'.format(flag) + '_' + "_ccsgld_Bias.txt", "w") as fp:
    fp.write("\n".join(str(item) for item in ccsgld_bias))

with open('{:02}'.format(flag) + '_' + "_csgld_Rmse.txt", "w") as fp:
    fp.write("\n".join(str(item) for item in csgld_rmse))

with open('{:02}'.format(flag) + '_' + "_ccsgld_Rmse.txt", "w") as fp:
    fp.write("\n".join(str(item) for item in ccsgld_rmse))

print('Job_', flag,  '_finished. \n')