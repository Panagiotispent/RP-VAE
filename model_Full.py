# -*- coding: utf-8 -*-

import torch
from torch import nn
import timeit
import torch.nn.functional as F


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
        
        self.corr_size = self.z_size * (self.z_size - 1) // 2
        
        # q
        self.q_mean = self._linear(self.feature_volume, self.z_size, relu=False)
        
        # lower diagonal elements with 2 unconstrained layers one corr, one variance
        self.q_logvar = self._linear(self.feature_volume, self.z_size, relu=False)
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

        mean, var_ltr, logcorr = self.q_full_cov(encoded) 
        
        lower_tri_cov, cov = self.lower_tri_cov(var_ltr,logcorr)
      
        z, cov = self.z_s(mean,cov)
        
        # for visualising the same batch 
        self.z = z.detach()
        
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        return (mean, lower_tri_cov, cov), x_reconstructed
    
    
    def time_forward(self,x):
        # encode x
        encoded = self.encoder(x)
        print('encoder()')
        print(min(timeit.repeat(lambda: self.encoder(x),globals=globals(),number= 100,repeat=10)))
        
        mean, var_ltr, logcorr = self.q_full_cov(encoded)
        print('q_full_cov()')
        print(min(timeit.repeat(lambda: self.q_full_cov(encoded),globals=globals(),number= 100,repeat=10)))
        
        lower_tri_matrix = self.lower_tri_cov(var_ltr,logcorr)
        print('lower_tri_cov()')
        print(min(timeit.repeat(lambda: self.lower_tri_cov(var_ltr,logcorr),globals=globals(),number= 100,repeat=10)))
      
        z = self.z_s(mean,lower_tri_matrix)
        print('z_s()')
        print(min(timeit.repeat(lambda: self.z_s(mean,lower_tri_matrix),globals=globals(),number= 100,repeat=10)))
        
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

   

    def q_full_cov(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled),self.q_logvar_corr(unrolled)
      

    def lower_tri_cov(self, log_var, corr):
        std = torch.exp(0.5 * log_var)  # diagonal Constrain
        batch_size = std.shape[0]
        dim = std.shape[-1]
        
        # Ensure the correlations are between -1 and 1
        corr = torch.tanh(corr)
    
        # Create a lower triangular matrix with softplus on the diagonal elements
        L = torch.zeros((batch_size, dim, dim), device=std.device)
        tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
        
        # Fill the lower triangular part with correlations
        L[:, tril_indices[0], tril_indices[1]] = corr
        
        # Apply softplus to the diagonal elements to ensure they are positive
        L[:, range(dim), range(dim)] = F.softplus(std)
        
        # Construct the covariance matrix as L * L^T
        lower_tri_cov = L @ L.transpose(1, 2)

        
        return L, lower_tri_cov

    def close_psd(self,cov):
        # Perform eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Set negative eigenvalues to a small positive value
        eigenvalues = torch.clamp(eigenvalues, min=1e-5)
        
        # Reconstruct the matrix
        psd_matrix = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)
        
        return psd_matrix

    
    def z_s(self,mean,cov):
        try:
            z = mean + torch.bmm(torch.linalg.cholesky(cov).float(),torch.randn([cov.shape[0],cov.shape[-1],1]).cuda()).squeeze(-1) # mean [bs,z]
        except:
            cov = self.close_psd(cov)
            z = mean + torch.bmm(torch.linalg.cholesky(cov).float(),torch.randn([cov.shape[0],cov.shape[-1],1]).cuda()).squeeze(-1) # mean [bs,z] 
        return z , cov
    
    
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    # # Random Projection kl_divergence
    def kl_divergence_loss(self, mean, tri,var): 
        
        l = 0
        mean = mean[:,:,None]
        
        # lvar = torch.bmm(tri,tri.transpose(2,1))
        # input(torch.min(torch.linalg.eigvalsh(lvar[0])))
        # input((torch.linalg.eigvalsh(lvar[0])))
        
        totrace = var + torch.bmm(mean, mean.transpose(2,1))

        l = 0.5 * (- torch.logdet(var) - mean.shape[-1]+ totrace.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1))

        return torch.mean(l) 
    
    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'Full'
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
