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
        self.cov_space = 10
        # self.RP_space = z_size
        
        
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
        # self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)
        
        # lower diagonal elements
        # self.q_logvar_ltr = self._linear(self.feature_volume, (self.cov_space + self.corr_size), relu=False)
        
        # lower diagonal elements with 2 unconstrained layers one corr, one variance
        self.q_logvar = self._linear(self.feature_volume, self.cov_space, relu=False)
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

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        # mean, logvar = self.q(encoded)
        
        # mean, var_ltr = self.q_full_cov(encoded) # mean, logvar_ltr,corr = self.q(encoded)
        
        mean, var_ltr, logcorr, lamda = self.q_full_cov(encoded) # mean, logvar_ltr,corr = self.q(encoded)
        
        lower_tri_matrix = self.lower_tri_cov(var_ltr,logcorr) # logvar_ltr, corr
      
        # lower_tri_matrix = self.lower_tri_cov(logvar_ltr) # logvar_ltr, corr  
        
        projected_ltr,projected_var = self.RPproject(lower_tri_matrix,lamda)
        
        # input(projected_ltr.shape)
        
        # tic = time.perf_counter()
        # toc = time.perf_counter()
        # print(f"{toc - tic:0.2f} seconds")

        
        # z = self.z_s(mean, logvar)
        # input(z.shape)
        # z = self.z_full_cov(mean,lower_tri_matrix)
        
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

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        # input(var_ltr.shape)
        return (mean, lower_tri_matrix, projected_var), x_reconstructed

    # ==============
    # VAE components
    # ==============

    # def q(self, encoded):
    #     unrolled = encoded.view(-1, self.feature_volume)
    #     return self.q_mean(unrolled), self.q_logvar(unrolled)
    
    # def q_full_cov(self, encoded):
    #     unrolled = encoded.view(-1, self.feature_volume)
    #     return self.q_mean(unrolled), self.q_logvar_ltr(unrolled)
    
    # # Using a two linear layers to generate 1.the diagonal and 2.correlation of the data for the lower_tri_matrix  
    # # '''https://github.com/boschresearch/unscented-autoencoder/blob/main/models/dist_utils.py'''
    def q_full_cov(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled),self.q_logvar_corr(unrolled),self.lamda(unrolled)
    
    # # # Using a single linear layer to generate both the diagonal and correlation of the data for the lower_tri_matrix    
    # # def lower_tri_cov(self,log_var):
    # #     dim = self.cov_space
    # #     tril_indices = torch.tril_indices(row=dim, col=dim)
        
    # #     lower_tri_matrix = torch.zeros([log_var.shape[0],dim,dim], device=log_var.device)
        
    # #     lower_tri_matrix[:, tril_indices[0], tril_indices[1]] = log_var
        
    # #     # input(lower_tri_matrix.shape)
        
    # #     return lower_tri_matrix
        
    
    # #   
    
    # # Unconstrained
    def lower_tri_cov(self,log_var,corr):
        
        # std = log_var.mul(0.5).exp_()
        batch_size = log_var.shape[0]
        dim = log_var.shape[-1]
     
        # build symmetric matrix with zeros on diagonal and correlations under 
        rho_matrix = torch.zeros((batch_size, dim, dim), device=corr.device)
        tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
        rho_matrix[:, tril_indices[0], tril_indices[1]] = corr 
        # input(rho_matrix[0])
        lower_tri_cov = rho_matrix
        
        lower_tri_cov[:,range(dim), range(dim)] = log_var 
       
        return lower_tri_cov
        
    
    # def z_s(self, mean, logvar):
    #     std = logvar.mul(0.5).exp_()
    #     eps = (
    #         Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
    #         Variable(torch.randn(std.size()))
    #     )
    #     return eps.mul(std).add_(mean)
    
    # def z_full_cov(self, mean, lower_tri_matrix):
        
    #     std = lower_tri_matrix.mul(0.5).exp_()
        
    #     eps = torch.randn(mean.shape[-1], device=mean.device)
        
    #     z = mean + std @ eps
        
    #     # input(z.shape)
    #     return z
    
    def RPproject(self,tri,lamda):
        
       
        g = torch.Generator(device=tri.device)
        
        # P = self.P # trainable projection matrix instead of random projection matrix for each datum
        random_samples = torch.zeros([tri.shape[0],self.z_size, self.cov_space],device=tri.device) # tri.shape >> z_size 
        P = torch.zeros([tri.shape[0],self.z_size, self.cov_space],device=tri.device)
        for i in range(tri.shape[0]):
            # set the seed for each datum
            # self.seed += 1 
            # g.manual_seed(self.seed)
            random_samples[i] = torch.randn(self.z_size, self.cov_space ,device=tri.device, generator = g) # .repeat(mean.shape[0], 1, 1) we need a fixed Projection matrix for each datum  so we can't use repeat
            # print(P.shape)
            
            (P[i],_) = torch.linalg.qr(random_samples[i])
        
        # random_samples = torch.randn(self.z_size, self.cov_space ,device=tri.device).repeat(tri.shape[0], 1, 1)
        # # input(random_samples.shape)
        # (P,_) = torch.linalg.qr(random_samples)
        # # input(P.shape)
        # input(tri.shape)
       
        sigma = torch.bmm(tri,tri.transpose(2,1)) 
        R = torch.bmm(P,sigma) # P @ tri [z,prj]
        RP = torch.bmm(R,P.transpose(2,1)) 
        # input(RP)
        log_lamda = lamda.exp() # contrain to positive the diagonal to be added to the projected covariance
        
        m_lamda = torch.diag_embed(log_lamda, offset=0, dim1=1)
        
        RP_var = RP + m_lamda
        
        # input(RP_var.shape)
        
        RP_tri = torch.bmm(P,tri)
        
        # l = torch.bmm(P,tri)
        
        # print((RP))
        # input((torch.bmm(l,l.transpose(2,1))))
        
        # input(- torch.logdet((torch.bmm(RP.transpose(2,1),RP))))
        return RP_tri, RP_var
    
    def z_RP(self,mean,var):
        #create the variance from tri @ tri.T
        # var = torch.bmm(tri,tri.transpose(2,1))
        # input(var.shape)
        # input(tri.transpose(2,1).shape)
        # input(tri.shape)
        # input((tri.transpose(2,1) @ torch.randn(tri.shape[-2]).cuda()).shape)
        z = mean + torch.tril(var,diagonal=0) @ torch.randn(var.shape[-1]).cuda() # mean [bs,z]
        # print( torch.tril(var,diagonal=0))
        # input(z)
        # reparameterization trick sampling 
        # z = torch.distributions.MultivariateNormal(mean,var).rsample()
       
        return z
    
    
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    # def kl_divergence_loss(self, mean, logvar):
    #     return ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)

    # def kl_divergence_loss(self, mean, logvar):
        
    #     # print(mean.shape)
    #     # print(logvar[0][0][0])
    #     logvar = torch.bmm(logvar,logvar.transpose(2,1)) # batch logvariance from the lower triangular matrix 
    #     # input(logvar[0][0][0])
    #     logvar_diag = torch.zeros(mean.shape, device=mean.device)
    #     for i in range(mean.shape[-1]):
    #         logvar_diag[:,i] =  logvar[:,i,i]
        
    #     # input(logvar_diag.shape)
    #     return ((mean**2 + logvar_diag.exp() - 1 - logvar_diag) / 2).sum() / mean.size(0)

    # # Random Projection kl_divergence
    def kl_divergence_loss(self, mean, tri,var): 
        ''' THIS IS CURRENTLY THE LOWER TRI MATRIX AND THE PROJECTED VARIANCE STILL GET A FEW NAN IN THE LOGDET'''
        l = 0
        lvar = torch.bmm(tri,tri.transpose(2,1)) # Sigma = Γ @ Γ.T
        # input(lvar)
        # input(torch.linalg.inv(lvar))
        # input(torch.logdet(lvar+ 1e-6))
        
        lvar = lvar + 1e-6
        # tic = time.perf_counter()
        # x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        # input(torch.linalg.inv(var))
        # reshape mean
        mean = mean[:,:,None]
        # input(- torch.logdet(lvar))
        totrace = var + torch.bmm(mean, mean.transpose(2,1))
        
        l = 0.5 * (- torch.logdet(lvar) - mean.shape[-1]+ totrace.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1))
        # input(torch.mean(l))
        # toc = time.perf_counter()
        # print(f"{toc - tic:0.2f} seconds")
        
        
        # print(l)
        
        # tic = time.perf_counter()
        
        # l = 0
        # for b in  range(mean.shape[0]):
        #     # input(mean[b])
        #     l += 0.5 * ( - torch.logdet(var[b]) - mean.shape[-1] + torch.trace(var[b] + mean[b]@ mean[b].T) ) 

        # input(l/mean.shape[0])
        # mean_l = l / mean.shape[0]
        # toc = time.perf_counter()
        # print(f"{toc - tic:0.2f} seconds")
        
        #batch mean l
        return torch.mean(l) 
    
    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'VAE'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size):
        z = self.get_z().cuda()if self._is_on_cuda() else self.get_z()
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
