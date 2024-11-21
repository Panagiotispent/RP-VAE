# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:58:06 2024

@author: panay
"""
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch import nn

import torch.nn.functional as F
import timeit

import bnaf
from torch.distributions import Normal, MultivariateNormal

# https://github.com/fmu2/flow-VAE/blob/master/dynamic_flow_vae.py#L15

''' dynamic flow '''
class PlanarFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of planar flow.

        Args:
            dim: input dimensionality.
        """
        super(PlanarFlow, self).__init__()

        self.linear_u = nn.Linear(dim, dim)
        self.linear_w = nn.Linear(dim, dim)
        self.linear_b = nn.Linear(dim, 1)

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        u, w, b = self.linear_u(v), self.linear_w(v), self.linear_b(v)

        def m(x):
            return F.softplus(x) - 1.
        def h(x):
            return torch.tanh(x)
        def h_prime(x):
            return 1. - h(x)**2

        inner = (w * u).sum(dim=1, keepdim=True)
        u = u + (m(inner) - inner) * w / (w * w).sum(dim=1, keepdim=True)
        activation = (w * x).sum(dim=1, keepdim=True) + b
        x = x + u * h(activation)
        psi = h_prime(activation) * w
        log_det = torch.log(torch.abs(1. + (u * psi).sum(dim=1, keepdim=True)))

        return x, v, log_det

class RadialFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of radial flow.

        Args:
            dim: input dimensionality.
        """
        super(RadialFlow, self).__init__()

        self.linear_a = nn.Linear(dim, 1)
        self.linear_b = nn.Linear(dim, 1)
        self.linear_c = nn.Linear(dim, dim)
        self.d = dim

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        a, b, c = self.linear_a(v), self.linear_b(v), self.linear_c(v)

        def m(x):
            return F.softplus(x)
        def h(r):
            return 1. / (a + r)
        def h_prime(r):
            return -h(r)**2

        a = torch.exp(a)
        b = -a + m(b)
        r = (x - c).norm(dim=1, keepdim=True)
        tmp = b * h(r)
        x = x + tmp * (x - c)
        log_det = (self.d - 1) * torch.log(1. + tmp) + torch.log(1. + tmp + b * h_prime(r) * r)

        return x, v, log_det

class HouseholderFlow(nn.Module):
    def __init__(self, dim):
        """Instantiates one step of householder flow.

        Args:
            dim: input dimensionality.
        """
        super(HouseholderFlow, self).__init__()

        self.linear_v = nn.Linear(dim, dim)

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        v = self.linear_v(v)
        [B, D] = list(v.size())
        outer = v.reshape(B, D, 1) * v.reshape(B, 1, D)
        v_sqr = (v * v).sum(dim=1)
        H = torch.eye(D).cuda() - 2 * outer / v_sqr.reshape(B, 1, 1)
        x = (H * x.reshape(B, 1, D)).sum(dim=2)
        
        return x, v, 0

class Flow(nn.Module):
    def __init__(self, dim, type, length):
        """Instantiates a chain of flows.

        Args:
            dim: input dimensionality.
            type: type of flow.
            length: length of flow.
        """
        super(Flow, self).__init__()

        if type == 'planar':
            self.flow = nn.ModuleList([PlanarFlow(dim) for _ in range(length)])
        elif type == 'radial':
            self.flow = nn.ModuleList([RadialFlow(dim) for _ in range(length)])
        elif type == 'householder':
            self.flow = nn.ModuleList([HouseholderFlow(dim) for _ in range(length)])
        else:
            self.flow = nn.ModuleList([])

    def forward(self, x, v):
        """Forward pass.

        Args:
            x: input tensor (B x D).
            v: output from last layer of encoder (B x D).
        Returns:
            transformed x and log-determinant of Jacobian.
        """
        [B, _] = list(x.size())
        log_det = torch.zeros(B, 1).cuda()
        for i in range(len(self.flow)):
            x, v, inc = self.flow[i](x, v)
            log_det = log_det + inc
 
        return x, log_det.squeeze(-1)

# BNAF model with custom functions
def create_bnaf_model(length,f_layers, z_size, n_size,res=''):
    '''
    Parameters
    ----------
    args : (flows,layers,hidden_dim,n_dim)
        length: number of flows
        f_layers: number of layers for each flow
        z_size: input features dimensions
        n_size: size of linear matrices
    
    Returns
    -------
    model : BNAF model
    '''

    flows = []
    for f in range(length):
        layers = []
        for _ in range(f_layers - 1):
            layers.append(
                bnaf.MaskedWeight(
                    z_size * n_size,
                    z_size * n_size,
                    dim=z_size,
                )
            )
            layers.append(bnaf.Tanh())

        flows.append(
            bnaf.BNAF(
                *(
                    [
                        bnaf.MaskedWeight(
                            z_size, z_size * n_size, dim=z_size
                        ),
                        bnaf.Tanh(),
                    ]
                    + layers
                    + [
                        bnaf.MaskedWeight(
                            z_size * n_size, z_size, dim=z_size
                        )
                    ]
                ),
                res=res if f < length - 1 else None
            )
        )

        if f < length - 1:
            flows.append(bnaf.Permutation(z_size, "flip"))

    model = bnaf.Sequential(*flows)

    return model


class VAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size, flow= '', length = 0, f_layers= 0, n_size = 0,res=''):
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
        
        self.n_flow = flow
        if flow == 'bnaf':
            self.flow = create_bnaf_model(length,f_layers, z_size, n_size,res)
        else:
            self.flow = Flow(z_size, flow, length)
            self.v = nn.Linear(z_size * 2, z_size) # dynamic flow type
        
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

        z, flow_det = self.z_s(mean, logvar)
        
        # for visualising the same batch 
        self.z = z.detach()
        
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        return (mean, logvar, flow_det), x_reconstructed
    
    
    def time_forward(self,x):

        # encode x
        encoded = self.encoder(x)
        print('encoder()')
        print(min(timeit.repeat(lambda: self.encoder(x),globals=globals(),number= 100,repeat=10)))
        
        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        print('q()')
        print(min(timeit.repeat(lambda: self.q(encoded),globals=globals(),number= 100,repeat=10)))
 
        z, flow_det = self.z_s(mean, logvar)
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
        z = eps.mul(std).add_(mean)
        
        if self.n_flow != 'bnaf':
            v = self.v(torch.cat((mean, logvar), dim=1))
            return self.flow(z, v)
        else:
            return self.flow(z)
        
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    # def kl_divergence_loss(self, mean, logvar, flow_log_det = 0 ):   
    #     logvar = torch.clamp(logvar, min=-10, max=10)
        
    #     input(((mean**2 + logvar.exp() - 1 - logvar) / 2).sum()/ mean.size(0) - flow_log_det )
    #     input((((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() - flow_log_det / mean.size(0)))
        
        
    #     return (((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() - flow_log_det / mean.size(0))

    def kl_divergence_loss(self, mu, logvar, log_det_jacobian_sum, clamp_value=10):
        
        # Numerical stability: Clamp the logvar to avoid very large or very small values
        logvar = torch.clamp(logvar, min=-clamp_value, max=clamp_value)

        # Base distribution is a standard Gaussian
        p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
        q = Normal(mu, torch.exp(0.5 * logvar))
        
        kl_div = torch.distributions.kl.kl_divergence(q, p).sum(-1)

        # Clamp the log determinant to a reasonable range
        log_det_jacobian_sum = torch.clamp(log_det_jacobian_sum, min=-clamp_value, max=clamp_value)

        kl_div -= log_det_jacobian_sum

        # Ensure KL divergence is non-negative
        kl_div = torch.clamp(kl_div, min=0)

        return kl_div.mean()

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
