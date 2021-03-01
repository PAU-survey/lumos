#!/usr/bin/env python

import torch
from torch import nn

class Lumos_model(nn.Module):
    def __init__(self):

        super().__init__()

        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=2,out_channels=32, kernel_size = 9, padding = 4),nn.MaxPool2d(2), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, padding = 4),nn.MaxPool2d(2), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(in_channels=64,out_channels=128, kernel_size = 5, padding = 3),nn.MaxPool2d(2), nn.BatchNorm2d(128), nn.ReLU(),
                                  nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding = 3),nn.MaxPool2d(2), nn.BatchNorm2d(64) ,nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, padding = 2),nn.MaxPool2d(2), nn.BatchNorm2d(32) ,nn.ReLU())

        self.f_ex1 = nn.Sequential(nn.Linear(32*4*4+11+2, 2000),nn.Dropout(0.02),nn.ReLU())
        self.f_ex2 = nn.Sequential(nn.Linear(2000, 1000),nn.Dropout(0.02),nn.ReLU())

        self.f_ex3 = nn.Sequential(nn.Linear(1000, 100),nn.Dropout(0.02),nn.ReLU())
        self.f_ex4a = nn.Sequential(nn.Linear(100, 5))
        self.f_ex4b = nn.Sequential(nn.Linear(100, 5))
        self.f_ex4c = nn.Sequential(nn.Linear(100, 5))

        self.embed = nn.Embedding(80, 10) #encoding the band in 10 parameters

        
        
    def forward(self, img,band,coord,Iauto):

        b = self.embed(band)

        x = self.encoder1(img)
        x = x.view(len(x),-1) #convert to 1D array

        #additional input information (magnitude, band, CCD galaxy coordinates)
        x = torch.cat([x,Iauto,b,coord],1)
        x = self.f_ex1(x)
        x = self.f_ex2(x)
        x = self.f_ex3(x)
        flux = self.f_ex4a(x)
        logalpha = self.f_ex4b(x)
        logsig = self.f_ex4c(x)

        logalpha = logalpha - torch.logsumexp(logalpha, 1)[:,None]


        return flux,logalpha,logsig
                                      


