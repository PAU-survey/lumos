#!/usr/bin/env python
# encoding: UTF8

# Force the CPU version to only use one thread. Needed for running
# at PIC, but also useful locally. There one can instead run multiple
# jobs in parallell.


import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import norm

from .lumos_network import Lumos_model
from .lumos import Lumos_net



def clear_catalogue(cat, alpha_keys, flux_keys, sigma_keys):
    cat.a1.where(cat.sigma_keys[0] < errlim, 0, inplace=True)
    cat.a2.where(cat.sigma_keys[1] < errlim, 0, inplace=True)
    cat.a3.where(cat.sigma_keys[2] < errlim, 0, inplace=True)
    cat.a4.where(cat.sigma_keys[3] < errlim, 0, inplace=True)
    cat.a5.where(cat.sigma_keys[4] < errlim, 0, inplace=True)
    cat = cat[cat[alpha_keys].sum(1) != 0]
    cat[alpha_keys] = cat[alpha_keys] / cat[alpha_keys].values.sum(1)[:,None]

    alphas, fluxes, sigmas = cat[alpha_keys], cat[flux_keys], cat[sigma_keys]
     
    return cat, alphas, fluxes, sigmas



def single_exposures_flux(alphas, fluxes, sigmas):
    flux = (alphas * fluxes).sum(1)
    sig = np.sqrt((alphas * sigmas**2).sum(1))

    return flux, sig


def coadds_flux_measurements(catcalib):
    refids = catcalib.red_id.unique().values

    grid0 = np.arange(-10,-2,0.1)
    grid1 = np.arange(-2,10,0.01)

    cat_coadds = pd.DataFrame()

    for gal in range(len(refids)):
        catsub = catcalib[(catcalib.ref_id == refids[gal])]
 
        flux_cal = catsub.zp.values[:,None]*catsub[flux_keys].values
        error_cal = np.sqrt(catsub.zp_error.values[:,None]**2*catsub[sigma_keys].values**2 + catsub.zp.values[:,None]**2*catsub[sigma_keys].values**2 + catsub.zp_error.values[:,None]**2*catsub[flux_keys].values**2)

        grid2 = np.arange(10,flux_cal.max()+200,0.1)
        grid = np.concatenate((grid0,grid1,grid2),0)


        pdfs =  catsub[alpha_keys].values[:,:,None] * norm.pdf(grid,loc = flux_cal[:,:,None],scale = error_cal[:,:,None])
        pdfs = pdfs.sum(1)
        pdfs_norm = pdfs / pdfs.sum(1)[:,None]

        cdf_norm = np.cumsum(pdfs_norm,1)


        pdfs_norm = pd.DataFrame(pdfs_norm)
        pdfs_norm['band'] = s.band.values

        gr = pdfs_norm.groupby(['band'])
        pdf_coadd = gr.prod()
        pdf_coadd = pdf_coadd / pdf_coadd.sum(1).values[:,None]
        cdf_coadd = np.cumsum(pdf_coadd.values,1)



        med_idx = np.abs(cdf_coadd - 0.5).argmin(1)
        f_medPDF = grid[med_idx]
        max_idx = np.argmax(pdf_coadd.values,1)
        f_maxPDF = grid[max_idx]

        q16_idx = np.abs(cdf_coadd - 0.16).argmin(1)
        q84_idx = np.abs(cdf_coadd - 0.84).argmin(1)
        q027_idx = np.abs(cdf_coadd - 0.027).argmin(1)
        q97_idx = np.abs(cdf_coadd - 0.97).argmin(1)

        q16 = grid[q16_idx]
        q84 = grid[q84_idx]
        q027 = grid[q027_idx]
        q97 = grid[q97_idx]

        sig68 = 0.5*(q84-q16)


        sub = pd.DataFrame(np.c_[refids[gal]*np.ones(shape = len(pdf_coadd)),pdf_coadd.index.values,f_maxPDF, sig68], columns = ['ref_id', 'band','flux', 'flux_error'])

        cat_coadds = pd.concat((cat_coadds,sub),0)

    return cat_coadds

        
