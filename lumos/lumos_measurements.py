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


class flux_measurements:
    """Measurement of the single and the coadded exposures"""
    
    def __init__(self, cat):
        """
        param cat: {DataFrame}: catalogue with the gaussian prediction by Lumos.
        """
        self.cat = cat
        self.a = ['a%s'%k for k in range(1,6)]
        self.f = ['f%s'%k for k in range(1,6)]
        self.s = ['e%s'%k for k in range(1,6)]
        
        self.ref_ids = cat.ref_id.unique()


    def single_exposures(self):
        flux = (self.cat[self.a].values * self.cat[self.f].values).sum(1)
        sig = np.sqrt((self.cat[self.a].values * self.cat[self.s].values**2).sum(1))
        
        
        cat_se = pd.DataFrame(np.c_[self.cat.ref_id, self.cat.image_id, flux, sig], columns = ['ref_id','image_id','flux','flux_error'])
        
        return cat_se
    
    def coadded_fluxes(self,zp,errlim =403):
        """
        param zp {DataFrame}: DataFrame with image_id, zp and zp_error    
        """
        catcalib = self.cat.merge(zp, on = 'image_id')
        catcalib = catcalib[catcalib.zp > 1]
        
        #define grid
        grid0 = np.arange(-10,-2,0.1)
        grid1 = np.arange(-2,10,0.01)
        
        ## correct very large errors
        catcalib.a1.where(catcalib.e1 < errlim, 0, inplace=True)
        catcalib.a2.where(catcalib.e2 < errlim, 0, inplace=True)
        catcalib.a3.where(catcalib.e3 < errlim, 0, inplace=True)
        catcalib.a4.where(catcalib.e4 < errlim, 0, inplace=True)
        catcalib.a5.where(catcalib.e5 < errlim, 0, inplace=True)
        catcalib = catcalib[catcalib[self.a].sum(1) != 0]
        catcalib[self.a] = catcalib[self.a] / catcalib[self.a].values.sum(1)[:,None]
        
        
        cat_coadds = pd.DataFrame()
        
        for gal in range(len(self.ref_ids)):
            s = catcalib[(catcalib.ref_id ==self.ref_ids[gal])]
            
            fcalib = s.zp.values[:,None]*s[self.f].values 
            ecalib = np.sqrt(s.zp_error.values[:,None]**2*s[self.s].values**2 + s.zp.values[:,None]**2*s[self.s].values**2 + s.zp_error.values[:,None]**2*s[self.f].values**2)
            
            
            fcalib_max = fcalib.max()
            
            #adequate grid for thies particular galaxy
            grid2 = np.arange(10,fcalib_max+200,0.1)
            grid = np.concatenate((grid0,grid1,grid2),0)

            
            pdfs =  s[self.a].values[:,:,None] * norm.pdf(grid,loc = fcalib[:,:,None],scale = ecalib[:,:,None])
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


            sub = pd.DataFrame(np.c_[self.ref_ids[gal]*np.ones(shape = len(pdf_coadd)),pdf_coadd.index.values,f_maxPDF, sig68], columns = ['ref_id', 'band','flux', 'flux_error'])

            cat_coadds = pd.concat((cat_coadds,sub),0)
            
        return cat_coadds



        
