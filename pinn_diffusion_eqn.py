# standard system modules
import os, sys

# standard module for array manipulation
import numpy as np

# standard module for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# standard research-level machine learning toolkit from Meta (FKA: FaceBook)
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    pass
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

import importlib

# Execute:
#
#   python monitor_losses.py losses.csv& 
#
# in a terminal window and the folder containing the file
# losses.csv to monitor the training 
import lossmonitor as lm
import flowde as fd
import pinndemod as pn
from pinndemod import SobolSample

from tqdm import tqdm

# update fonts
FONTSIZE = 12
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=False)

# set a seed to ensure reproducibility
#seed = 314159
#rnd  = np.random.RandomState(seed)

#----------------------------------------------------------------------

# set device to cuda if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device: {str(DEVICE):4s}')

fd.plot_pdf(fd.X0, fd.fdist)

# convert to a tensor of shape (M, d) with d = 1
X0 = torch.Tensor(fd.X0).view(-1, 1).to(DEVICE)

X0.shape

#----------------------------------------------------------------------

ZMIN =-4.0
ZMAX = 4.0
# domain of PINN
#           t,     z
LOWER = [ 0.0,  ZMIN]
UPPER = [ 1.0,  ZMAX]

print('LOWER, UPPER')
print(LOWER)
print(UPPER)
print()

# training constants
PEXP       = 17                     # DATA_SIZE = 2**PEXP
SIGMA0     = 1e-3                   # see preamble above

VAL_SIZE   = 5000                   # size of validation sample
BATCH_SIZE = 2048
NUM_ITERS  = 10_000                 # number of training iterations
MC_SIZE    = 4096                   # sample size, {x_0^(n)} n=1,..,N

# check whether to load model parameters
if 'PINN_LOAD' in os.environ:
    LOAD = eval(os.environ['PINN_LOAD'])
else:
    LOAD = False

# number of learning rate steps
if 'PINN_NUM_STEPS' in os.environ:
    NUM_STEPS = eval(os.environ['PINN_NUM_STEPS'])
else:
    NUM_STEPS = 2
    
# initial learning rate
if 'PINN_BASE_LR' in os.environ:
    BASE_LR = eval(os.environ['PINN_BASE_LR'])
else:
    BASE_LR = 4.e-4
    
# learning rate scale factor
if 'PINN_GAMMA' in os.environ:
    GAMMA = eval(os.environ['PINN_GAMMA'])
    print('scale factor evaluated')
else:
    GAMMA = 0.5

# compute validation loss after the following number of iterations
MONITOR_STEP  = 20

# sample size of test data
TEST_SIZE = 10000

# output files
TIMELEFT_FILE = 'timeleft.txt'
LOSS_FILE     = 'pinn1d_losses.csv'
PARAMS_FILE   = 'pinn1d_params.pth'
RESULTS_FILE  = 'pinn1d_results.png'

# If True usethe  Limited-memory Broyden–Fletcher–Goldfarb–Shanno optimization
# algorithm, otherwise use Adam

USE_LBFGS = True

TRAIN = True

#----------------------------------------------------------------------

import scipy

N = 25          # number of flowlines from t=1 to t=0.

SAVEPATH = True # save flowlines if True

# instantiate flowline finite-difference (F-D) solver
flow = fd.FlowDE(X0, T=400, sigma0=SIGMA0, savepath=SAVEPATH, step=2)

# generate z values at quantiles of the normal distribution
p = np.linspace(0, 1, N+2)[1:-1] # probability values
z = torch.Tensor(scipy.stats.norm.ppf(p)).view(-1, 1).to(DEVICE)

# compute flowlines with F-D solver
t, x = flow(z)

#----------------------------------------------------------------------

K =10 #the amount of time values we care to look at
#A = torch.gather(x, 0, torch.tensor(IT[:K])

I = np.arange(0, len(t), 1) 
np.random.shuffle(I)
#gives us a random list from 0-199 so which we can match with our t-values/arrays of the x-tensor

J = np.arange(0, len(z), 1)
np.random.shuffle(J)
#gives us a random list from 0-24 so we can match it with our z-values/elements in the arrays of the x-tensor

L = x[I[:K]] #shortens our tensor into K randomly chosen arrays
T = t[I[:K]] #shortens our t array into K randomly chosen elements
Z = z[J[:K]] #shortens our z array into K randomly chosen elements

X = L[:,J[:K]] #shortens the length of our arrays in the remaining x tensor into K elements that match our remaining z array

dfx = pd.DataFrame(X).T

dft = pd.concat([pd.DataFrame(T)]*10, axis=1, ignore_index=True).T

dfz = pd.concat([pd.DataFrame(Z)]*10, axis=1, ignore_index=True)

npx = dfx.to_numpy()
npt = dft.to_numpy()
npz = dfz.to_numpy()

tzx = np.stack([npx, npt, npz], axis=2)
TZX = tzx.reshape(-1, 3)

#----------------------------------------------------------------------

class ObjODE(nn.Module):

    def __init__(self, x0, solution, sigma0=1e-3, alpha=1.0, debug=False):
        super().__init__()

        assert(x0.ndim==2)
        # x0.shape: (M, d)

        # cache solution object
        self.solution = solution
        self.sigma0 = sigma0
        self.alpha = alpha
        self.debug = debug
        self.q = fd.qVectorField(x0, sigma0)
        
    def eval(self):
        self.solution.eval()

    def train(self):
        self.solution.train()

    def save(self, paramsfile):
        self.solution.save(paramsfile)
        
    def forward(self, t, z):

        # compute solution with current free parameters
        
        x = self.solution(t, z)
        
        # compute dx/dt
        #
        # autograd.grad computes the gradient of scalars, but since
        # x(t, z) is generally a batch of scalars with shape 
        # (batch-size, 1), we need to tell autograd.grad to apply 
        # the gradient operator to every scalar in the batch. We do 
        # this by passing autograd.grad a tensor of shape (batch-size, 1). 
        # It is also necessary to create the computation graph for 
        # this operation. 
      
        dxdt = torch.autograd.grad(x, t, 
                                   torch.ones_like(x), 
                                   create_graph=True)[0]

        losses = (self.q.sigma(t) * dxdt - self.alpha * self.q.Gprime(t, x))**2
        
        return torch.mean(losses)

#----------------------------------------------------------------------

class ObjData(nn.Module):

    def __init__(self, solution, data, debug=False):
        super().__init__()

        # cache solution object
        self.solution = solution
        D = torch.Tensor(data).to(DEVICE)
        self.t = D[:, 1].view(-1, 1)
        self.z = D[:, 2].view(-1, 1)
        self.x = D[:, 0].view(-1, 1)
        self.debug = debug
        
    def eval(self):
        self.solution.eval()

    def train(self):
        self.solution.train()

    def save(self, paramsfile):
        self.solution.save(paramsfile)
        
    def forward(self):

        # compute solution with current free parameters
        
        xpred = self.solution(self.t, self.z)

        losses = (xpred-self.x)**2
        
        return torch.mean(losses)

#----------------------------------------------------------------------

# create raw data using Sobol sampling
rawdata = pn.SobolSample(LOWER, UPPER, num_points_exp=PEXP)

DATA_SIZE = len(rawdata)
TRAIN_SIZE = DATA_SIZE - VAL_SIZE   # size of training sample

rawdata.shape

#----------------------------------------------------------------------



#----------------------------------------------------------------------
# Dataset for training
print('\n\ttrain_dataset')
train_dataset = pn.PINNDataset(rawdata,
                               start=0, end=TRAIN_SIZE, 
                               device=DEVICE)

# Dataset for computing average loss, after a number of iterations,
# of size VAL_SIZE that overlaps with train_dataset
print('\n\ttrain_small_dataset')
train_small_dataset  = pn.PINNDataset(rawdata, 
                                      start=0, end=TRAIN_SIZE,
                                      random_sample_size=VAL_SIZE,
                                      device=DEVICE)

# Dataset, independent of train_dataset, for monitoring training 
print('\n\tval_dataset')
val_dataset   = pn.PINNDataset(rawdata, 
                               start=TRAIN_SIZE, end=TRAIN_SIZE+VAL_SIZE,
                               device=DEVICE)

#----------------------------------------------------------------------

# Loader for training data
train_loader = pn.PINNDataLoader(train_dataset,
                                 batch_size=BATCH_SIZE, 
                                 num_iterations=NUM_ITERS)
print()

#train_loader_pytorch = DataLoader(
#        train_dataset,
#        sampler= BatchSampler(RandomSampler(train_dataset),
#                             batch_size=BATCH_SIZE, 
#                             drop_last=True))
print()

# Loader for random subset of training data of size VAL_SIZE
train_small_loader = pn.PINNDataLoader(train_small_dataset, batch_size=VAL_SIZE)
print()

# Loader for validation data
val_loader = pn.PINNDataLoader(val_dataset, batch_size=VAL_SIZE)

#----------------------------------------------------------------------

def pinnsolver(objective, optimizer, scheduler, 
               train_loader, train_small_loader, val_loader, 
          lossfile='losses.csv',
          timeleftfile='timeleft.txt',
          paramsfile='net.pth',
          step=500, 
               frac=0.015, 
               delete=True):

    # this is needed by the LBFGS optimizer
    def closure_(t, z):
        # clear all gradients
        optimizer.zero_grad()

        # compute average loss
        y = objective(t, z)

        # compute gradients
        y.backward(retain_graph=True)

        return y

    # instantiate object that saves average losses to a csv file 
    # for realtime monitoring

    number_iterations = len(train_loader)
    
    losswriter = lm.LossWriter(number_iterations, 
                               lossfile, timeleftfile, 
                               step=step, 
                               frac=frac, 
                               delete=delete,
                               model=objective, 
                               paramsfile=paramsfile) 
    
    # training loop
    
    current_lr = -1.0

    use_Adam = str(type(optimizer)).find('Adam') > 0
    
    for ii, (t, z) in enumerate(train_loader):

        # set mode to training so that training-specific 
        # operations such as dropout, etc., are enabled.
        objective.train()

        if use_Adam:
            # clear all gradients
            optimizer.zero_grad()

            # compute average loss
            y = objective(t, z)

            # compute gradients
            y.backward()

            # take one step downhill in the average loss landscape
            optimizer.step()
            
        else:  
            # LBFGS needs the closure() function, without an argument, 
            # so use lambda-function one-liner to make this function
            # on-the-fly.
            optimizer.step(lambda: closure_(t, z))

        scheduler.step()
        
        # i'm alive printout
        if (ii % step == 0) or (ii == number_iterations-1):

            objective.eval()
            
            t_loss = pn.compute_avg_loss(objective, train_small_loader)
            
            v_loss = pn.compute_avg_loss(objective, val_loader)

            # update loss file

            lr = optimizer.param_groups[0]['lr']

            if lr != current_lr:
                current_lr = lr
                print()
                print(f'\t\tlearning rate: {lr:10.3e}')
                
            losswriter(ii, t_loss, v_loss, lr)

#----------------------------------------------------------------------

# instantiate neural network
net = pn.FCN(n_inputs=2, n_hidden=8, n_width=32).to(DEVICE)

# instantiate solution ansatz
soln = pn.Solution(net).to(DEVICE)
LOAD = False ##CHANGE LATER
if LOAD:
    print('\nLOAD parameters from', PARAMS_FILE)
    print()
    soln.load(PARAMS_FILE)
    
# instantiate function to be minimized
objective = pn.Objective(ObjODE(X0[:MC_SIZE], soln), ObjData(soln, TZX), alpha).to(DEVICE)
# ---------------------------------------------------------------------
print(net)
print()
print(f'number of parameters: {pn.number_of_parameters(net)}')

#----------------------------------------------------------------------

# Number of milestones in multistep LR schedule
n_milestones = NUM_STEPS - 1
print(f'number of milestones: {n_milestones:10d}\n')

# Learning rate iteration milestones
iters_per_milestones = int(len(train_loader) / NUM_STEPS)
milestones = [n * iters_per_milestones for n in range(NUM_STEPS)]
learning_rates = [BASE_LR * GAMMA**i for i in range(NUM_STEPS)]

print("Step | Milestone |      LR")
print("--------------------------")
for i in range(NUM_STEPS):
    print(f"{i:>4} | {milestones[i]:>9} | "
          f"{learning_rates[i]:<10.1e}")
    if i < 1:
        print("--------------------------")

# drop first entry of milestones list because it contains the base LR
milestones = milestones[1:]

#----------------------------------------------------------------------

if TRAIN:
    if USE_LBFGS:
        print('\n\tUse LBFGS optimizer\n')
        optimizer = torch.optim.LBFGS(soln.parameters(),
                                          lr=BASE_LR,
                                          # max_iter=1000,
                                          # max_eval=100,
                                          # history_size=100,
                                          # tolerance_grad=1e-20,
                                          # tolerance_change=1e-20,
                                          line_search_fn="strong_wolfe")
    else:
        print('\n\tUse Adam optimizer\n')
        optimizer = torch.optim.Adam(soln.parameters(), lr=BASE_LR)
    
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=GAMMA)
    
    pinnsolver(objective, optimizer, scheduler,
               train_loader, train_small_loader, val_loader, 
               lossfile=LOSS_FILE,
               timeleftfile=TIMELEFT_FILE,
               paramsfile=PARAMS_FILE,
               step=MONITOR_STEP)

#----------------------------------------------------------------------

# create raw data using Sobol sequence

testdata = pn.UniformSample(LOWER, UPPER, num_points=TEST_SIZE)
print()

# Dataset for training
print('\n\ttest_dataset')
test_dataset = pn.PINNDataset(testdata,
                               start=0, end=TEST_SIZE, 
                               device=DEVICE)
print()

# Loader for training data
test_loader = pn.PINNDataLoader(test_dataset,
                                batch_size=TEST_SIZE)
print()

avloss = pn.compute_avg_loss(objective, test_loader)
print(f'average loss on test data: {avloss:10.3e}')

#----------------------------------------------------------------------

import scipy

N = 25          # number of flowlines from t=1 to t=0.

SAVEPATH = True # save flowlines if True

# instantiate flowline finite-difference (F-D) solver
flow = fd.FlowDE(X0, T=400, sigma0=SIGMA0, savepath=SAVEPATH, step=2)

# generate z values at quantiles of the normal distribution
p = np.linspace(0, 1, N+2)[1:-1] # probability values
z = torch.Tensor(scipy.stats.norm.ppf(p)).view(-1, 1).to(DEVICE)

# compute flowlines with F-D solver
t, x = flow(z)

t = t.to(DEVICE)
x = x.to(DEVICE)

X = []
for j, z in enumerate(x[0]):
    X.append(x[:, j].view(-1, 1))
    X.append(soln(t, z))
X = torch.stack(X, dim=1).squeeze(-1)

fd.plot_flows((t, X), 
              filename=RESULTS_FILE,
              linewidth=[1.5, 1], 
              linestyle=['solid', 'dashed'],
              color=['blue', 'red'])

#----------------------------------------------------------------------