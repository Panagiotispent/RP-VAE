# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch import nn
import timeit

class VAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size,n_size = 0):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        # #for visualisation reasons
        self.z = torch.zeros(10,self.z_size)

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num, last=True),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)
        
        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)
        
        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num, last=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)

        z = self.z_s(mean, logvar)
        
        # for visualising the same batch 
        self.z = z.detach()
        
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        return (mean, logvar, None), x_reconstructed
    
    
    def time_forward(self,x,Pr):
        # encode x
        encoded = self.encoder(x)
        print('encoder()')
        print(min(timeit.repeat(lambda: self.encoder(x),globals=globals(),number= 100,repeat=10)))
        
        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        print('q()')
        print(min(timeit.repeat(lambda: self.q(encoded),globals=globals(),number= 100,repeat=10)))
 
        z = self.z_s(mean, logvar)
        print('z_s()')
        print(min(timeit.repeat(lambda: self.z_s(mean, logvar),globals=globals(),number= 100,repeat=10)))
        
        # for visualising the same batch 
        self.z = z.detach()
        
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        print('project(z).view()')
        print(min(timeit.repeat(lambda:  self.project(z).view(-1, self.kernel_num,self.feature_size, self.feature_size,),globals=globals(),number= 100,repeat=10)))  

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)
        print('decoder()')
        print(min(timeit.repeat(lambda: self.decoder(z_projected),globals=globals(),number= 100,repeat=10)))  

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z_s(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)
        
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar ,var =None):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'D'
            '-{cov}'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            cov = self.z_size,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size):
        z = self.get_z().cuda()if self._is_on_cuda() else self.get_z()
        z = z[:size] # Required to be less than batch size
        
        # To sample from the latent space directly instead of ground truth
        # z = Variable(
        #     torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
        #     torch.randn(size, self.z_size)
        # )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data
    
    def get_z(self):
        return self.z
    
    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num, last=False):
        conv = nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=3, stride=2, padding=1,
        )
        return conv if last else nn.Sequential(
            conv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num, last=False):
        deconv = nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        )
        return deconv if last else nn.Sequential(
            deconv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)
