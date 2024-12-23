import torch
import numpy as np
import random
from scipy.linalg import sqrtm
from sklearn import datasets

def sample_normal(size=64, loc=(0, 0), scale=(0.2, 0.2), device='cpu'):
    return np.random.normal(size=(size, 2), loc=loc, scale=scale).astype(np.float32)

def sample_mixture(size=128, loc0=(0, -1), loc1=(0, 1), 
                   scale=(0.2, 0.2), weights=(0.25, 0.75), device='cpu'):
    locs = np.array([loc0, loc1])
    indices = random.choices(range(len(locs)), k=size, weights=weights)
    balls = locs[indices] + np.random.normal(size=(size, 2), loc=(0, 0), scale=scale).astype(np.float32)
    return balls

def sample_mixture3(size=128, loc0=(0, -1), loc1=(0, 0), loc2=(0, 1),
                   scale=(0.2, 0.2), weights=(0.25, 0.75, 0.25), device='cpu'):
    locs = np.array([loc0, loc1, loc2])
    indices = random.choices(range(len(locs)), k=size, weights=weights)
    balls = locs[indices] + np.random.normal(size=(size, 2), loc=(0, 0), scale=scale).astype(np.float32)
    return balls

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
    
class MixtureNormalSampler(Sampler):
    def __init__(self, dim=2, loc0=(0, -1), loc1=(0, 1), 
                   scale=(0.2, 0.2), weights=(0.25, 0.75),device='cpu'):
        super(MixtureNormalSampler, self).__init__(device=device)
        self.dim = dim
        self.loc0, self.loc1 = loc0, loc1
        self.scale = scale
        self.weights = weights
        device = self.device
        
    def sample(self, batch_size=10):
        batch = sample_mixture(size=batch_size, loc0=self.loc0, loc1=self.loc1, 
                   scale=self.scale, weights=self.weights, device=self.device)
        return torch.tensor(batch, device=self.device)
    
class MixtureNormalSampler_outliers(Sampler):
    def __init__(self, dim=2, loc0=(0, -1), loc1=(0, 1), loc2=(0,2),
                   scale=(0.2, 0.2), weights=(0.25, 0.74, 0.01),device='cpu'):
        super(MixtureNormalSampler_outliers, self).__init__(device=device)
        self.dim = dim
        self.loc0, self.loc1, self.loc2 = loc0, loc1, loc2
        self.scale = scale
        self.weights = weights
        device = self.device
        
    def sample(self, batch_size=10):
        batch = sample_mixture3(size=batch_size, loc0=self.loc0, loc1=self.loc1, loc2=self.loc2,
                   scale=self.scale, weights=self.weights, device=self.device)
        return torch.tensor(batch, device=self.device)
    

    
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        assert size <= self.loader.batch_size
        try:
            batch, _ = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)
            
        return batch[:size].to(self.device)
    

class TensorSampler(Sampler):
    def __init__(self, tensor, device='cuda'):
        super(TensorSampler, self).__init__(device)
        self.tensor = torch.clone(tensor).to(device)
        
    def sample(self, size=5):
        assert size <= self.tensor.shape[0]
        
        ind = torch.tensor(np.random.choice(np.arange(self.tensor.shape[0]), size=size, replace=False), device=self.device)
        return torch.clone(self.tensor[ind]).detach().to(self.device)    
    
