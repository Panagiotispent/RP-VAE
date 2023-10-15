# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:00:33 2023

@author: panay
"""
import torch
from torch.autograd import Variable
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F
import time

class VAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        
        self.seed = 0
        self.cov_space = z_size//10
        
        
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
        
        self.corr_size = self.cov_space * (self.cov_space - 1) // 2
        
        # q
        self.q_mean = self._linear(self.feature_volume, self.z_size, relu=False)
        
        # lower diagonal elements with 2 unconstrained layers one corr, one variance
        self.q_logvar = self._linear(self.feature_volume, self.cov_space, relu=False)
        self.q_logvar_corr = self._linear(self.feature_volume, (self.corr_size), relu=False)
       
        
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

        mean, var_ltr, logcorr = self.q_full_cov(encoded) # mean, logvar_ltr,corr = self.q(encoded)
        
        lower_tri_matrix = self.lower_tri_cov(var_ltr,logcorr) # logvar_ltr, corr
      
        projected_ltr,projected_var = self.RPproject(lower_tri_matrix)

        z = self.z_RP(mean,projected_ltr)
        
        # for visualising the same batch 
        self.z = z.detach()
        
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        return (mean, lower_tri_matrix, projected_var), x_reconstructed

    # ==============
    # VAE components
    # ==============

   
    # # Using a two linear layers to generate 1.the diagonal and 2.correlation of the data for the lower_tri_matrix  
    # # '''https://github.com/boschresearch/unscented-autoencoder/blob/main/models/dist_utils.py'''
    def q_full_cov(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled),self.q_logvar_corr(unrolled)
    
    
    # # # Unconstrained
    # def lower_tri_cov(self,log_var,corr):
        
    #     # std = log_var.mul(0.5).exp_()
    #     batch_size = log_var.shape[0]
    #     dim = log_var.shape[-1]
     
    #     # build symmetric matrix with zeros on diagonal and correlations under 
    #     rho_matrix = torch.zeros((batch_size, dim, dim), device=corr.device)
    #     tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
    #     rho_matrix[:, tril_indices[0], tril_indices[1]] = corr 
    #     lower_tri_cov = rho_matrix
        
    #     lower_tri_cov[:,range(dim), range(dim)] = log_var 
       
    #     return lower_tri_cov
    
    #constrained 
    def lower_tri_cov(self, log_var,corr):
        std = torch.exp(0.5 * log_var)
        batch_size = std.shape[0]
        dim = std.shape[-1]
        rho_dim = corr.shape[-1]
         
        # build symmetric matrix with sigma_x * sigma_x on the diagonal and sigma_x * sigma_y off-diagonal
        var_matrix = std.unsqueeze(-1).repeat(1, 1, dim)
        var_matrix = var_matrix * var_matrix.transpose(2, 1)  # el-wise product
         
        # build symmetric matrix with zeros on diagonal and correlations under
        rho_matrix = torch.zeros((batch_size, dim, dim), device=std.device)
        tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
        rho_matrix[:, tril_indices[0], tril_indices[1]] = corr
         
        # build lower triangular covariance
        lower_tri_cov = var_matrix * rho_matrix  # multiply correlations and std's
        lower_tri_cov = lower_tri_cov + torch.diag_embed(std)+ 1e-6  # add std's on diagonal
        
        return lower_tri_cov
 
    
    
    def RPproject(self,tri):
        # Fixed sampling
        g = torch.Generator(device=tri.device)

        random_samples = torch.zeros([tri.shape[0],self.z_size, self.cov_space],device=tri.device) # tri.shape >> z_size 
        P = torch.zeros([tri.shape[0],self.z_size, self.cov_space],device=tri.device)
        for i in range(tri.shape[0]):
            random_samples[i] = torch.randn(self.z_size, self.cov_space ,device=tri.device, generator = g) # .repeat(mean.shape[0], 1, 1) we need a fixed Projection matrix for each datum  so we can't use repeat
            
            (P[i],_) = torch.linalg.qr(random_samples[i])
       
        # Random sampling
        # random_samples = torch.randn(self.z_size, self.cov_space ,device=tri.device).repeat(tri.shape[0], 1, 1)
        # (P,_) = torch.linalg.qr(random_samples)

        sigma = torch.bmm(tri,tri.transpose(2,1)) 
        R = torch.bmm(P,sigma) # P @ tri [z,prj]
        RP_var = torch.bmm(R,P.transpose(2,1)) 
        
        RP_tri = torch.bmm(P,tri)
        
        return  RP_tri,RP_var
    
    def z_RP(self,mean,ltr):
        z = mean + ltr @ torch.randn(ltr.shape[-1]).cuda() # mean [bs,z]
        return z
    
    
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    # # Random Projection kl_divergence
    def kl_divergence_loss(self, mean, tri,var): 
        ''' THIS IS CURRENTLY THE LOWER TRI MATRIX AND THE PROJECTED VARIANCE STILL GET A FEW NAN IN THE LOGDET'''
        l = 0
        mean = mean[:,:,None]
        
        lvar = torch.bmm(tri,tri.transpose(2,1)) # Sigma = Γ @ Γ.T
        # lvar = lvar + 1e-6
        totrace = var + torch.bmm(mean, mean.transpose(2,1))

        l = 0.5 * (- torch.logdet(lvar) - mean.shape[-1]+ totrace.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1))
       
        return torch.mean(l) 
    
    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'RPVAE'
            '-{z}'
            '-{cov}'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            z = self.z_size,
            cov = self.cov_space,
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
