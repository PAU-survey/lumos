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
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, dataset, TensorDataset
from torch import nn, optim
from pathlib import Path


from .lumos_network import Lumos_model
from lumos.data_set import DataSet


from astropy.modeling.functional_models import Sersic2D
from astropy.modeling.functional_models import Moffat2D

from scipy.signal import convolve, convolve2d, fftconvolve
from skimage.measure import block_reduce

from scipy.special import erf


class Lumos_train:
    """Interface for photometry prection using neural networks."""
    
    # Here we estimate photometry on CPUs. This should be much
    # simpler to integrate and sufficiently fast.
    def __init__(self, data_dir, epochs,model_outdir, nsamps,  model_path = None, pretrained = False, batch_size=500, stamp_shape=(60, 60)):
        
        # Load the model.
        cnn = Lumos_model()


        
        self.batch_size = batch_size
        self.cnn = cnn.cuda()
        
        self.data_dir = data_dir
        self.stamp_shape = stamp_shape
        self.epochs = epochs
        self.model_outdir = model_outdir
        self.pretrained = pretrained
        self.nsamps = nsamps
        
        if self.pretrained == True:
            cnn.load_state_dict(torch.load(model_path, map_location='cpu'))
            cnn.eval()
       
                                         
    def _train(self):
    # Training the network.

        optimizer = optim.Adam(self.cnn.parameters(), lr=1e-4) #, weight_decay=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        data_dir = Path(self.data_dir)
        dset = DataSet(data_dir, self.nsamps)

        train_size = int(0.9 * len(dset))
        test_size = len(dset) - train_size
    
        dset_train, dset_test = torch.utils.data.random_split(dset, [train_size, test_size])
    
        loader_train = DataLoader(dset_train, batch_size=500, shuffle = True)
        loader_test = DataLoader(dset_test, batch_size=500)

        
        for epoch in range(self.epochs):
            print('epoch: ', epoch)
                
            for meta, prof_conv, stamp, field in loader_train:
            
                meta = meta.squeeze(1)
            
                #this depends on the order the metadata parameters are stored
                n, psf, lab, x0,y0, Iauto, band = meta[:,1].unsqueeze(1), meta[:,2].unsqueeze(1),meta[:,3].unsqueeze(1),meta[:,4].unsqueeze(1),meta[:,5].unsqueeze(1), meta[:,10].unsqueeze(1), meta[:,11]

                stamp = torch.cat((stamp.unsqueeze(1), prof_conv.unsqueeze(1)), 1)
                meta = meta.float().squeeze(1)
                

                coord = torch.Tensor(np.c_[x0,y0])
                band = torch.LongTensor(band.numpy())

                optimizer.zero_grad()
                stamp = stamp.cuda().float()

                flux, logalpha,logsig = self.cnn(stamp,  band.cuda(),coord.cuda(),Iauto.cuda())

                logsig = torch.clamp(logsig,-6,6)
                sig = torch.exp(logsig)

                logerf = torch.log(erf(lab.cpu()/(np.sqrt(2)*sig.detach().cpu())+1))

                log_prob_truncated =   logalpha - 0.5*(flux - lab.cuda()).pow(2) / sig.pow(2) - logsig - logerf.cuda()

                field = field.cuda()
                log_prob =  log_prob_truncated 
                log_prob = torch.logsumexp(log_prob, 1)
                loss = -log_prob.mean()


                loss.backward()
                optimizer.step()

            scheduler.step()

            loss = loss.detach().cpu().numpy()
        
            if epoch % 5:
            
                torch.save(self.cnn.state_dict(), self.model_outdir)
                continue
            

        return self.cnn

