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

from .lumos_network import Lumos_model

from astropy.modeling.functional_models import Sersic2D
from astropy.modeling.functional_models import Moffat2D

from scipy.signal import convolve, convolve2d, fftconvolve
from skimage.measure import block_reduce


class Lumos_net:
    """Interface for photometry prection using neural networks."""
    
    # Here we estimate photometry on CPUs. This should be much
    # simpler to integrate and sufficiently fast.
    def __init__(self, model_path, batch_size=500):
        
        # Load the model.
        cnn = Lumos_model()
        cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
        cnn.eval()
       
        self.batch_size = batch_size
        self.cnn = cnn
   
    def _internal_naming(self, band, intervention):
        """Converting to internal band numbering."""
        band = str(band) + '_' + str(intervention)
        
        print(band)
        # Convention based on how the bands are laid out in the trays.
       
        D = {'NB455_0': 1,'NB465_0': 2,'NB475_0': 3,'NB485_0': 4, 'NB495_0': 5, 'NB505_0': 6, 'NB515_0': 7, 'NB525_0': 8, \
             'NB535_0': 9, 'NB545_0': 10, 'NB555_0': 11, 'NB565_0': 12, 'NB575_0': 13, 'NB585_0': 14, 'NB595_0': 15, \
             'NB605_0': 16, 'NB615_0': 24, 'NB625_0': 23, 'NB635_0': 22, 'NB645_0': 21, 'NB655_0': 20, 'NB665_0': 19, \
             'NB675_0': 18, 'NB685_0': 17, 'NB695_0': 32, 'NB705_0': 31, 'NB715_0': 30, 'NB725_0': 29, 'NB735_0': 28, \
             'NB745_0': 27,'NB755_0': 26, 'NB765_0': 25, 'NB775_0': 40, 'NB785_0': 39, 'NB795_0': 38, 'NB805_0': 37, \
             'NB815_0': 36, 'NB825_0': 35, 'NB835_0': 34, 'NB845_0': 33, \
             'NB455_1': 41,'NB465_1': 42,'NB475_1': 43,'NB485_1': 44, 'NB495_1': 45, 'NB505_1': 46, 'NB515_1': 47, 'NB525_1': 48, \
             'NB535_1': 49, 'NB545_1': 50, 'NB555_1': 51, 'NB565_1': 52, 'NB575_1': 53, 'NB585_1': 54, 'NB595_1': 55, \
             'NB605_1': 56, 'NB615_1': 64, 'NB625_1': 63, 'NB635_1': 62, 'NB645_1': 61, 'NB655_1': 60, 'NB665_1': 59, \
             'NB675_1': 58, 'NB685_1': 57, 'NB695_1': 72, 'NB705_1': 71, 'NB715_1': 70, 'NB725_1': 69, 'NB735_1': 68, \
             'NB745_1': 67,'NB755_1': 66, 'NB765_1': 65, 'NB775_1': 80, 'NB785_1': 79, 'NB795_1': 78, 'NB805_1': 77, \
             'NB815_1': 76, 'NB825_1': 75, 'NB835_1': 74, 'NB845_1': 73}
        nr = D[band] - 1    

        return nr

    def create_cutouts(self, img, coord_pix):
        """Create the cutouts from positions given in pixels."""

        npos = len(coord_pix)
        cutouts = np.zeros(shape=(npos, 60, 60), dtype = np.float32)

        L = []
        for i, (ind, sub) in enumerate(coord_pix.iterrows()):
            # Remember to make a copy of the array.
            cutout = img[int(np.round(sub.x,0))-30:int(np.round(sub.x,0))+30, int(np.round(sub.y,0))-30:int(np.round(sub.y,0))+30].copy()
            cutout = cutout.astype(np.float32, copy = False)
            
            dx, dy = np.round(10*(sub.x - np.round(sub.x,0))), np.round(10*(sub.y - np.round(sub.y,0)))
            cutouts[i] = cutout
                    
        return cutouts
      
      
    def create_modelled_profiles(self, metadata):
        """Create the cutouts from positions given in pixels."""

        alph = 4.765
        nprofs = len(metadata)
        profiles = np.zeros(shape=(nprofs, 60, 60), dtype = np.float32)
        xgrid,ygrid = np.meshgrid(np.arange(0,600,1), np.arange(0,600,1))
        xgrid_psf,ygrid_psf = np.meshgrid(np.arange(0,400,1), np.arange(0,400,1))
        
        for i in range(nprofs):
            #cosmos_pixelscale = 0.03, paus_pixelscale = 0.263, draw profile in a x10 resolution grid
            r50, n, ellip, psf, Iauto, x, y, theta = 10*metadata.r50 * 0.03 / 0.263, metadata.sersic_n_gim2d, 1 - metadata.aperture_b/metadata.aperture_a,\
            10* metadata.psf_fwhm /0.263, metadata.I_auto, metadata.aperture_y, metadata.aperture_x,  (180 - metadata.aperture_theta)*(2*np.pi/360)
            dx, dy = np.round(10*(metadata.aperture_y - np.round(metadata.aperture_y,0))), np.round(10*(metadata.aperture_x - np.round(metadata.aperture_x,0)))
            
            #create the galaxy profile: 
            mod = Sersic2D(amplitude = 1, r_eff =r50[i], n=n[i], x_0=300+dx[i], y_0=300+dy[i],theta =  theta[i], ellip = ellip[i])
            prof =  mod(xgrid, ygrid)
            
            #create the PSF profile: 
            gam = psf[i] / (2. * np.sqrt(np.power(2., 1 / alph ) - 1.))
            amp = (alph - 1) / (np.pi * gam**2)
            moff = Moffat2D(amplitude=amp, x_0 = 200, y_0 = 200, gamma=gam, alpha=alph)
            prof_psf = moff(xgrid_psf, ygrid_psf)
            
            #convolve the profile, reduce to pixel resolution and normalise it
            prof_conv = fftconvolve(prof,prof_psf, mode = 'same')
            prof_conv = block_reduce(prof_conv, (10,10), np.mean)
            prof_conv = prof_conv / prof_conv.max()

            profiles[i] = prof_conv
     
        return profiles
    
    def _asdataset(self, cutouts, profiles, metadata):
        """Convert to a dataset."""

        cutouts = torch.tensor(cutouts).unsqueeze(1)
        profiles = torch.tensor(profiles).unsqueeze(1)
        
        net_inp_img = torch.cat((cutouts,profiles),1)

                
        add_input = metadata.astype(np.float32, copy=False)
        coord = torch.Tensor(np.c_[metadata.aperture_y.values, metadata.aperture_x.values]).unsqueeze(1)
        band = torch.LongTensor(metadata.band.values).unsqueeze(1)
        I_auto = torch.Tensor(metadata.I_auto.values).unsqueeze(1)
        
     
        dset = TensorDataset(net_inp_img, band,coord, I_auto)
        return dset
    
    def _photometry_cutouts(self, net_inp_img, net_inp_prof,metadata):
        """Determine the bakground for the postage stamps."""
        
        dset  = self._asdataset(net_inp_img, net_inp_prof, metadata)
        loader = DataLoader(dset, batch_size=self.batch_size, \
                            shuffle=False)

        pred = []
        for bstamp, bband,bcoord,bIauto in loader:

            with torch.no_grad():
                flux,logalpha,logsig = self.cnn(bstamp, bband,bcoord,bIauto)
                             
            flux = flux.detach().numpy()
            logalpha = logalpha.detach().numpy()
            logsig = logsig.detach().numpy()
            alpha = np.exp(logalpha) / np.exp(logalpha).sum(1)[:,None]
                            
            fluxerr = np.exp(logsig)
                             
            df_pred = pd.DataFrame(np.c_[alpha,flux,fluxerr], columns = ['a1','a2','a3','a4','a5','f1','f2','f3','f4','f5','e1','e2','e3','e4','e5'])
                             

        return df_pred

    def _photometry_img(self, img, coords_pix, metadata):
        """Predict the photometry using Lumos."""

        stamps = self.create_cutouts(img, coords_pix)
        profiles = self.create_modelled_profiles(metadata)
        interv = metadata.interv.values[0]
        metadata['band'] = self._internal_naming(metadata.band.values[0], interv) #*np.ones(shape = len(metadata))
                             
        pred = self._photometry_cutouts(stamps,profiles, metadata)

        return pred

    
