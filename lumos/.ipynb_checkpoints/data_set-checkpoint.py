#!/usr/bin/env python
# encoding: UTF(

import numpy as np
from glob import glob
import torch


class DataSet:
    """Custom dataset loader."""

    def __init__(self, data_dir, nsamps, stamp_shape=(60, 60)):
        self.data_dir = data_dir
        self.stamp_shape = stamp_shape
        self.nsamps = nsamps

    def __len__(self):
        ngal = self.nsamps#len(glob(str(self.data_dir / 'data_*')))

        return ngal

    def _load_matrix(self, pre, i):
        path = str(self.data_dir / f'data_{i}' / f'{pre}_{i}.npy')
        stamp = np.load(path).reshape(self.stamp_shape)
        stamp = np.nan_to_num(stamp)
        #stamp = stamp[30:90,30:90]

        stamp = torch.FloatTensor(stamp)

        return stamp

    def _load_matrix_bkg(self, pre, i):
        path = str(self.data_dir / f'data_{i}' / f'{pre}_{i}.npy')
        stamp = np.load(path)#.reshape((240,120))
        #stamp = stamp[30:90,30:90]
        stamp = torch.FloatTensor(stamp)

        return stamp

    def __getitem__(self, i):


        #print(i)
        path_meta = str(self.data_dir / f'data_{i}' / f'metadata_{i}.npy')
        #the metadata should include (at least) the iband mag, the CCD cooridnates.
        meta = torch.FloatTensor(np.nan_to_num(np.load(path_meta)))

        #out metadata file contains the iband magnitude in the 10th position
        Iauto = meta[:,10]
        field = torch.Tensor([0 if mag < 21.5 else 1 for mag in Iauto] ) 

        stamp = self._load_matrix('stamp', i)
        prof_conv = self._load_matrix('prof_conv', i)
        prof_conv = prof_conv / prof_conv.max()

        flip = [np.random.choice([0,1])]

        prof_conv = torch.flip(prof_conv, flip)
        stamp = torch.flip(stamp,flip)


        return meta, prof_conv, stamp, field