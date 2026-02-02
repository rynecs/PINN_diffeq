# ----------------------------------------------------------------------------
# # GSOC PINNDE Project
# > Sijil Jose, Pushpalatha C. Bhat, Sergei Gleyzer, Harrison B. Prosper<br>
# > Created: Tue May 27 2025<br>
# > Updated: Thu May 29 2025 HBP: generalize sigma(t)<br>
# > Updated: Sat Jul 19 2025 HBP: inspired by Sijil's logsumexp suggestion, 
#                                 try using torch.softmax to compute weights.
# ----------------------------------------------------------------------------
# standard system modules
import os, sys, re

# standard module for array manipulation
import numpy as np

# standard research-level machine learning toolkit from Meta (FKA: FaceBook)
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from flowde import qVectorField

import scipy.stats as st
# ----------------------------------------------------------------------------
def number_of_parameters(model):
    '''
    Get number of trainable parameters in a model.
    '''
    return sum(param.numel() 
               for param in model.parameters() 
               if param.requires_grad)

# This function assumes that the len(loader) is the same as
# the batch size given when the loader is instantiated
def compute_avg_loss(objective, loader):    
    assert(len(loader)==1)

    for phi, init_conds in loader:
        # Detach from computation tree and send to CPU (if on a GPU)
        avg_loss = float(objective(phi, init_conds).detach().cpu())
        
    return avg_loss
# ----------------------------------------------------------------------------
class Sin(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
        
class FCN(nn.Module):
    '''
    Model a fully-connected network (FCN).
    '''
    
    def __init__(self, 
                 n_inputs=2, 
                 n_hidden=4, 
                 n_width=32, 
                 nonlinearity=Sin):
        
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_width  = n_width
        
        cmd  = 'nn.Sequential(nn.Linear(n_inputs, n_width), nonlinearity(), '
        cmd += ', '.join(['nn.Linear(n_width, n_width), nonlinearity()' 
                          for _ in range(n_hidden-1)])
        cmd += ', nn.Linear(n_width, 1))'
        
        self.net = eval(cmd)

    def save(self, dictfile):
        # save parameters of neural network
        torch.save(self.state_dict(), dictfile)

    def load(self, dictfile):
        # load parameters of neural network and set to eval mode
        self.load_state_dict(torch.load(dictfile, 
                                        weights_only=True,
                                        map_location=torch.device('cpu')))
        self.eval()

    def forward(self, t, z):
        assert(t.ndim==2)
        assert(z.ndim==2)
        y = torch.concat((t, z), dim=-1)
        y = self.net(y)   
        return y
# ----------------------------------------------------------------------------
# Solution ansatz
# ----------------------------------------------------------------------------
class Solution(nn.Module):
    '''
    Model solution, using Theory of Connections
    '''
    def __init__(self, NN):

        # remember to initialize base (that is, parent) class
        super().__init__()

        # cache neural network and send to computational device
        self.g = NN
    
    def save(self, dictfile):
        # save parameters of neural network
        torch.save(self.g.state_dict(), dictfile)

    def load(self, dictfile):
        # load parameters of neural network and set to eval mode
        self.g.load_state_dict(torch.load(dictfile, 
                                          weights_only=True,
                                          map_location=torch.device('cpu')))

    def forward(self, t, z):
        t = t.view(-1, 1)
        if z.ndim < 2: z = z.repeat(len(t), 1)
            
        g = self.g

        ones = torch.ones_like(t)

        # ansatz            
        x = z + g(t, z) - g(ones, z)

        return x

    def diff(self, x, t):
        # compute dx/dt
        dxdt = torch.autograd.grad(x, t, 
                                   torch.ones_like(t), 
                                   create_graph=True)[0]
        return dxdt
# ----------------------------------------------------------------------------
# Objective function:
#  losses = E[(sigma(t) * dx/dt - alpha * Gprime(t, x))**2)]
# ----------------------------------------------------------------------------
class Objective(nn.Module):

    def __init__(self, objode, objdata, weight, debug=False):
        super().__init__()

        # cache solution object
        self.objode = objode
        self.objdata = objdata
        self.weight = weight
        self.debug = debug
        
    def eval(self):
        self.objode.solution.eval()

    def train(self):
        self.objode.solution.train()

    def save(self, paramsfile):
        self.objode.solution.save(paramsfile)
        
    def forward(self, t, z):

        # compute solution with current free parameters
        
        obj1 = self.objode(t, z)
        obj2 = self.objdata()
        
        return (1-self.weight)*obj1 + (self.weight)*obj2
# ----------------------------------------------------------------------------
# Using a Sobol sequence to created a sample of points
# ----------------------------------------------------------------------------
class SobolSample(np.ndarray):
    def __new__(cls,
                 lower_bounds,
                 upper_bounds,
                 num_points_exp=17, # of points = 2^num_points_exp
                 verbose=1):
       
        # Generate Sobol points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        sampler = st.qmc.Sobol(d=D, scramble=True)
        sample  = sampler.random_base2(m=num_points_exp) 
        sample  = st.qmc.scale(sample, lower_bounds, upper_bounds)

        if verbose:
            print("SobolSample")
            print(f"  {2**num_points_exp} Sobol points created in unit {D}-cube.")

        # Cast the numpy array to the type SobolSample
        sample = np.asarray(sample).view(cls)
        return sample
# ----------------------------------------------------------------------------
# Use uniform sampling to create a sample of points
# ----------------------------------------------------------------------------
class UniformSample(np.ndarray):
    def __new__(cls,
                 lower_bounds,
                 upper_bounds,
                 num_points,   # of points
                 verbose=1):

        # Generate points in the unit D-cube and scale to bounds
        D = len(lower_bounds)
        sample = np.random.uniform(0, 1, D*num_points).reshape((num_points, D))
        sample = st.qmc.scale(sample, lower_bounds, upper_bounds)
        
        if verbose:
            print("UniformSample")
            print(f"  {num_points} uniformly sampled points created in unit {D}-cube.")

        # Cast the numpy array to the type UniformSample
        sample = np.asarray(sample).view(cls)
        return sample
# ---------------------------------------------------------------------------
# Custom Dataset that takes (N, D) array of N points in the unit D-cube,
# where D=3. Taken from AIMS PINN project
# ---------------------------------------------------------------------------
class PINNDataset(Dataset):
    def __init__(self, data, start, end, 
                 random_sample_size=None,
                 device=torch.device("cpu"),
                 verbose=1):
        
        super().__init__()
        print('PINNDatset:type(data)', type(data))
        self.verbose = verbose

        # # Check that we have the right data types
        # if not isinstance(data, (SobolSample, UniformSample)):
        #     raise TypeError('''
        #     The object at argument 1 must be an instance of SobolSample
        #     or UniformSample
        #     ''')

        if random_sample_size == None:
            tdata = torch.Tensor(data[start:end])
        else:
            # Create a random sample from items in the specified range (start, end)
            assert(type(random_sample_size) == type(0))
            
            length  = end - start
            assert(length > 0)
            
            indices = torch.randint(0, length-1, size=(random_sample_size,))
            tdata   = torch.Tensor(data[indices])

        self.t = tdata[:, 0].reshape(-1, 1).requires_grad_().to(device)
        self.z = tdata[:, 1:].to(device)

        if verbose:
            print('PINNDataset')
            print(f"  shape of t: {self.t.shape}")
            print(f"  shape of z: {self.z.shape}")
    
    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):           
        return self.t[idx], self.z[idx]
# ---------------------------------------------------------------------------
# Custom DataLoader
# ---------------------------------------------------------------------------
class PINNDataLoader:
    '''
    A simple data loader that is much faster than the default PyTorch DataLoader.
    
    Notes:
    
    1. If used, sampler must be a PyTorch sampler and the arguments
       batch_size, shuffle, num_iterations are ignored.
       
    2. If num_iterations is specified, it is assumed that this is the
       desired maximum number of iterations, maxiter, per for-loop. 
       The flag shuffle is set to True and an internal count, defined by
       shuffle_step = floor(len(dataset) / batch_size) is computed. 
       The indices for accessing items from the dataset are shuffled every
       time the following condition is True

           itnum % shuffle_step == 0,

       where itnum is an internal counter that keeps track of the number of
       iterations. If num_iterations is not specified (the default), then
       the maximum number of iterations, maxiter = shuffle_step.
       
       This dataloader does not provide the option to return the last batch 
       if the latter is shorter than batch_size.
    '''
    def __init__(self, dataset, 
                 batch_size=None,
                 num_iterations=None,
                 sampler=None,
                 verbose=1,
                 debug=0,
                 shuffle=False):

        # Note: sampler and (batch_size, shuffle, num_iterations) are 
        # mutually exclusive
        self.dataset = dataset
        self.size    = batch_size
        self.niterations = num_iterations
        self.sampler = sampler
        self.verbose = verbose
        self.debug   = debug
        self.shuffle = shuffle
        
        if self.sampler:
            # Assume we're using a PyTorch sampler
            self.maxiter = len(self.sampler)
            self.size    = sampler.batch_size
            self.shuffle = False # we defer to the sampler
            self.shuffle_step = 1
            
            if self.verbose:
                print('PINNDataLoader')
                print('  ** Sampler specified')
                print(f'  dataset size: {len(self.dataset):10d}')
                print(f'  maxiter:      {self.maxiter:10d}')
                print(f'  batch_size:   {self.size:10d}')
        else:
            # Not using a sampler, so need batch_size
            if self.size == None:
                raise ValueError("Since you're not using a PyTorch sampler, "\
                                 "you must specify a batch_size")
                
            # If shuffle, then shuffle the dataset every shuffle_step iterations
            self.shuffle_step = int(len(dataset) / self.size)

            if self.niterations != None:
                # The maximum number of iterations has been specified
                assert(type(self.niterations)==type(0))
                assert(self.niterations > 0)
                
                self.maxiter = self.niterations
                
                # IMPORTANT: shuffle indices every self.shuffle_step iterations
                self.shuffle = True

                if self.verbose:
                    print('PINNDataLoader')
                    print('  ** Number of iterations specified')
                    print(f'  dataset size: {len(self.dataset):10d}')
                    print(f'  maxiter:      {self.maxiter:10d}')
                    print(f'  batch_size:   {self.size:10d}')
                    print(f'  shuffle_step: {self.shuffle_step:10d}')
                    
            elif len(dataset) > self.size:
                self.maxiter = self.shuffle_step
                
                if self.verbose:
                    print('PINNDataLoader')
                    print(f'  dataset size: {len(self.dataset):10d}')
                    print(f'  maxiter:      {self.maxiter:10d}')
                    print(f'  batch_size:   {self.size:10d}')
                    print(f'  shuffle_step: {self.shuffle_step:10d}')
  
            else:
                # Note: this could be = 2 for a 2-tuple of tensors!
                self.size = len(dataset)
                self.shuffle_step = 1
                self.maxiter = self.shuffle_step
                
                if self.verbose:
                    print('PINNDataLoader')
                    print(f'  dataset size: {len(self.dataset):10d}')
                    print(f'  maxiter:      {self.maxiter:10d}')
                    print(f'  batch_size:   {self.size:10d}')
                    print(f'  shuffle_step: {self.shuffle_step:10d}')

        assert(self.maxiter > 0)

        # initialize iteration number
        # IMPORTANT: must start at -1 so that itnum goes from
        # 0 to size - 1 # Thank you Lydia (July 21, 2025)
        self.itnum = -1

    # Tell Python to make objects of type EZDataLoader iterable
    def __iter__(self):
        return self

    # This method implements and terminates iterations
    def __next__(self): 

        # IMPORTANT: increment iteration number here!
        self.itnum += 1

        if self.itnum < self.maxiter:

            if self.sampler:
                # Create a new tensor indexing dataset using the sequence
                # returned by the sampler
                indices = list(sampler)[0]
                return self.dataset[indices]

            elif self.shuffle:
                # Create a new tensor indexing dataset via a random
                # sequence of indices
                jtnum = self.itnum % self.shuffle_step
                if jtnum == 0:
                    if self.debug > 0:
                        print(f'PINNDataLoader/shuffle indices @ ({self.itnum}, {jtnum})')
      
                    self.indices = torch.randperm(len(self.dataset))

                start = jtnum * self.size
                end = start + self.size
                indices = self.indices[start:end]
                return self.dataset[indices]
                
            else:
                # Create a new tensor directly indexing dataset
                start = self.itnum * self.size
                end = start + self.size
                return self.dataset[start:end]
        else:
            # Terminate iteration and reset iteration counter
            # IMPORTANT: must start at -1 so that itnum goes from
            # 0 to size - 1
            self.itnum = -1
            raise StopIteration

    def __len__(self):
        return self.maxiter