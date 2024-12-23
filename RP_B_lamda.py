# -*- coding: utf-8 -*-

import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        
        self.cov_space =  z_size//10
        
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
        self.q_logvar = self._linear(self.feature_volume, (self.cov_space), relu=False)
        self.q_logvar_corr = self._linear(self.feature_volume, (self.corr_size), relu=False)
        self.lamda = self._linear(self.feature_volume, self.z_size, relu=False)
        
        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num, last=True),
            nn.Sigmoid()
        )

    def forward(self, x,Pr):
        # encode x
        encoded = self.encoder(x)
        
        mean, var_ltr, logcorr,lamda = self.q_full_cov(encoded) 
        
        
        lower_tri_matrix = self.lower_tri_cov(var_ltr,logcorr) # logvar_ltr, corr
            

        projected_var = self.RPproject(lower_tri_matrix,lamda,Pr)
        
        z = self.z_RP(mean,projected_var)
        
        # for visualising the same batch 
        self.z = z.detach()
        # print(self.z)
        
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
    
    
    def q_full_cov(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled),self.q_logvar_corr(unrolled), self.lamda(unrolled)
    
    # # # Unconstrained
    # def lower_tri_cov_un(self,log_var,corr):
        
    #     # std = log_var.mul(0.5).exp_()
    #     batch_size = log_var.shape[0]
    #     dim = log_var.shape[-1]
     
    #     # build symmetric matrix with zeros on diagonal and correlations under 
    #     rho_matrix = torch.zeros((batch_size, dim, dim), device=corr.device)
    #     tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
    #     rho_matrix[:, tril_indices[0], tril_indices[1]] = corr 
    #     # input(rho_matrix[0])
    #     lower_tri_cov = rho_matrix
        
    #     lower_tri_cov[:,range(dim), range(dim)] = log_var 
       
    #     return lower_tri_cov
    
    # # '''https://github.com/boschresearch/unscented-autoencoder/blob/main/models/dist_utils.py'''
    # Constrained 
    def lower_tri_cov(self, log_var,corr):
        std = torch.exp(0.5 * log_var)
        batch_size = std.shape[0]
        dim = std.shape[-1]
         
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
    
    def RPproject(self,tri,lamda,p):
        
        
        # RP_tri= torch.diag_embed(lamda).to(tri.device)
        
        # to not double the values of the diagonal when adding the diagonal later
        RP_tri= torch.zeros_like(torch.diag_embed(lamda)).to(tri.device) 
        
        
        # torch.set_printoptions(threshold=10_000)
        # input(RP_tri[0])
        pos_end = p + tri.shape[1]
                
        # a LOT faster
        mask = torch.zeros_like(RP_tri).bool()
        for i in range(RP_tri.shape[0]):
            mask[i,int(p[i]):int(pos_end[i]),int(p[i]):int(pos_end[i])] = True

        RP_tri_mask = RP_tri.masked_scatter_(mask, tri)
        # input(RP_tri_mask.shape)
        
        # RP_var = torch.bmm(RP_tri_mask,RP_tri_mask.transpose(2,1)) 
        # # torch.set_printoptions(threshold=10_000)
        # # input(pos)
        # # input(RP_tri[0])
        
        
        ''' Random cov block projected each iteration'''
        
        # # RP_tri= torch.diag_embed(lamda).to(tri.device)
        # max_pos = (self.z_size - self.cov_space)

        
        # # a LOT faster
        # mask = torch.zeros_like(RP_tri).bool()
        # for i in range(RP_tri.shape[0]):
        #     p = torch.randint(0, max_pos,(1,)) 
            
        #     pos_end = p + tri.shape[1]
                
        #     mask[i,p:pos_end,p:pos_end] = True

        # RP_tri_mask = RP_tri.masked_scatter_(mask, tri)
        
        
        # RP_tri_mask[:,range(RP_tri_mask.shape[1]),range(RP_tri_mask.shape[2])] = lamda.exp() # change back the diag values and constrain
        
        #Change or add #???
        RP_tri_mask[:,range(RP_tri_mask.shape[1]),range(RP_tri_mask.shape[2])] += lamda # add the diag values 
        # input(RP_tri_mask[0])
        
        RP_var = torch.bmm(RP_tri_mask,RP_tri_mask.transpose(2,1)) 
        # input(RP_var[0])
        # torch.set_printoptions(threshold=10_000)
        # input(pos)
        # input(RP_var[0])
        
        return  RP_var
    
    
    
    # def RPproject(self,tri,lamda,P):
    #     P = P.to(tri.device)
    #     sigma = torch.bmm(tri,tri.transpose(2,1)) 
    #     R = torch.bmm(P,sigma) # P @ tri [z,prj]
    #     RP = torch.bmm(R,P.transpose(2,1)) 
        
    #     log_lamda = lamda.exp() # contrain to positive the diagonal to be added to the projected covariance
        
    #     m_lamda = torch.diag_embed(log_lamda, offset=0, dim1=1)
        
    #     RP_var = RP + m_lamda
        
    #     return RP_var
    
    def z_RP(self,mean,var):
        z = mean + torch.tril(var,diagonal=0) @ torch.randn(var.shape[-1]).cuda() # mean [bs,z]
        return z
    
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)
    
    # # Random Projection kl_divergence
    def kl_divergence_loss(self, mean, tri,var): 

        lvar = torch.bmm(tri,tri.transpose(2,1))

        mean = mean[:,:,None]

        totrace = var + torch.bmm(mean, mean.transpose(2,1))
        
        l = 0.5 * (- torch.logdet(lvar) - mean.shape[-1]+ totrace.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1))

        #batch mean l
        return torch.mean(l) 
    
    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'RP_B_D'
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
