import pdb
import math

from torch import nn

from torch.nn.functional import softmax, log_softmax
import torch
import geotorch

from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

from tqdm import tqdm

class ULightOT(nn.Module):
    def __init__(self, dim=2, k_potentials=5, l_potentials=5, epsilon=1, is_diagonal=True,
                 sampling_batch_size=1, S_diagonal_init=0.1, Sigma_diagonal_init=0.1):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.k_potentials = k_potentials
        self.l_potentials = l_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size
        
        self.log_alpha_raw = nn.Parameter(self.epsilon*torch.log(torch.ones(k_potentials)/k_potentials))
        self.log_beta_raw = nn.Parameter(self.epsilon*torch.log(torch.ones(l_potentials)/l_potentials))
        self.r = nn.Parameter(torch.randn(k_potentials, dim))
        self.mu = nn.Parameter(torch.randn(l_potentials, dim))
        
        self.S_log_diagonal_matrix = nn.Parameter(torch.log(S_diagonal_init*torch.ones(k_potentials, self.dim)))
        self.S_rotation_matrix = nn.Parameter(
            torch.randn(k_potentials, self.dim, self.dim)
        )
        
        self.Sigma_log_diagonal_matrix = nn.Parameter(torch.log(Sigma_diagonal_init*torch.ones(l_potentials, self.dim)))
        self.Sigma_rotation_matrix = nn.Parameter(
            torch.randn(l_potentials, self.dim, self.dim)
        )
        
        geotorch.orthogonal(self, "S_rotation_matrix")
        geotorch.orthogonal(self, "Sigma_rotation_matrix")

    def init_r_by_samples(self, samples):
        assert samples.shape[0] == self.r.shape[0]
        
        self.r.data = torch.clone(samples.to(self.r.device))
        
    def init_mu_by_samples(self, samples):
        assert samples.shape[0] == self.mu.shape[0]
        
        self.mu.data = torch.clone(samples.to(self.mu.device))
    
    def get_S(self):
        if self.is_diagonal:
            S = torch.exp(self.S_log_diagonal_matrix)
        else:
            S = (self.S_rotation_matrix*(torch.exp(self.S_log_diagonal_matrix))[:, None, :])@torch.permute(self.S_rotation_matrix, (0, 2, 1))
        return S
    
    def get_Sigma(self):
        if self.is_diagonal:
            Sigma = torch.exp(self.Sigma_log_diagonal_matrix)
        else:
            Sigma = (self.Sigma_rotation_matrix*(torch.exp(self.Sigma_log_diagonal_matrix))[:, None, :])@torch.permute(self.Sigma_rotation_matrix, (0, 2, 1))
        return Sigma
    
    def get_r(self):
        return self.r
    
    def get_mu(self):
        return self.mu
    
    def get_log_alpha(self):
        return (1/self.epsilon)*self.log_alpha_raw # ask nikita
    
    def get_log_beta(self):
        return (1/self.epsilon)*self.log_beta_raw

    def get_potential(self, x): # unnormalized density
        S = self.get_S()
        r = self.get_r()
        log_alpha = self.get_log_alpha()
        
        epsilon = self.epsilon
        
        if self.is_diagonal:
            mix = Categorical(logits=log_alpha)
            comp = Independent(Normal(loc=r, scale=torch.sqrt(self.epsilon*S)), 1)
            gmm = MixtureSameFamily(mix, comp)
            
            # density is normalized by default --> add logsumexp 
            potential = gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)
        else:
            mix = Categorical(logits=log_alpha)
            comp = MultivariateNormal(loc=r, covariance_matrix=self.epsilon*S)
            gmm = MixtureSameFamily(mix, comp)
            
            # density is normalized by default --> add logsumexp 
            potential = gmm.log_prob(x) + torch.logsumexp(log_alpha, dim=-1)
        
        return potential
    
    
    def get_C(self, x):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.get_log_alpha()
        
        eps_S = epsilon*S
        
        if self.is_diagonal:
            x_S_x = (x[:, None, :]*S[None, :, :]*x[:, None, :]).sum(dim=-1)
            x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
        else:
            x_S_x = (x[:, None, None, :]@(S[None, :, :, :]@x[:, None, :, None]))[:, :, 0, 0]
            x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
            
        exp_argument = (x_S_x + 2*x_r)/(2*epsilon) + log_alpha[None, :]
        log_norm_const = torch.logsumexp(exp_argument, dim=-1)
        
        return log_norm_const
    
    def get_marginal(self, x): # normalized density
        Sigma = self.get_Sigma()
        mu = self.get_mu()
        log_beta = self.get_log_beta()
        
        epsilon = self.epsilon
        
        if self.is_diagonal:
            mix = Categorical(logits=log_beta)
            comp = Independent(Normal(loc=mu, scale=torch.sqrt(self.epsilon*Sigma)), 1)
            gmm = MixtureSameFamily(mix, comp)
            
            potential = gmm.log_prob(x)
        else:
            mix = Categorical(logits=log_beta)
            comp = MultivariateNormal(loc=mu, covariance_matrix=self.epsilon*Sigma)
            gmm = MixtureSameFamily(mix, comp)
            
            potential = gmm.log_prob(x)
        
        return potential
    
    def sample_marginal(self, size): # normalized density
        Sigma = self.get_Sigma()
        mu = self.get_mu()
        log_beta = self.get_log_beta()
        samples = []
        
        epsilon = self.epsilon
        
        if self.is_diagonal:
            mix = Categorical(logits=log_beta)
            comp = Independent(Normal(loc=mu, scale=torch.sqrt(self.epsilon*Sigma)), 1)
            gmm = MixtureSameFamily(mix, comp)
        else:
            mix = Categorical(logits=log_beta)
            comp = MultivariateNormal(loc=mu, covariance_matrix=self.epsilon*Sigma)
            gmm = MixtureSameFamily(mix, comp)
            
        for i in range(size):
            samples.append(gmm.sample()[None, :])

        samples = torch.cat(samples, dim=0)
        
        return samples
    
    
    def set_epsilon(self, new_epsilon):
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)
    
        
    @torch.no_grad()
    def forward(self, x):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        
        log_alpha = self.get_log_alpha()
        
        eps_S = epsilon*S
            
        samples = []
        batch_size = x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size//sampling_batch_size if batch_size % sampling_batch_size == 0 else (batch_size//sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = x[sampling_batch_size*i:sampling_batch_size*(i+1)]
            
            if self.is_diagonal:
                x_S_x = (sub_batch_x[:, None, :]*S[None, :, :]*sub_batch_x[:, None, :]).sum(dim=-1)
                x_r = (sub_batch_x[:, None, :]*r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + S[None, :]*sub_batch_x[:, None, :]
            else:
                x_S_x = (sub_batch_x[:, None, None, :]@(S[None, :, :, :]@sub_batch_x[:, None, :, None]))[:, :, 0, 0]
                x_r = (sub_batch_x[:, None, :]*r[None, :, :]).sum(dim=-1)
                r_x = r[None, :, :] + (S[None, :, : , :]@sub_batch_x[:, None, :, None])[:, :, :, 0]
                
            exp_argument = (x_S_x + 2*x_r)/(2*epsilon) + log_alpha[None, :]
            
            if self.is_diagonal:                
                mix = Categorical(logits=exp_argument)
                comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon*S)[None, :, :]), 1)
                gmm = MixtureSameFamily(mix, comp) 
            else:
                mix = Categorical(logits=exp_argument)
                comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon*S)
                gmm = MixtureSameFamily(mix, comp)

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)

        return samples