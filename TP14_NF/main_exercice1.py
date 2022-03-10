import torch

from affine import Affine
from utils import FlowModel
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import torch.distributions as dists

# Model parameters 
nb_flow = 1
epochs = 10000
d = 3
mu = torch.Tensor([0, 10, 3])
sigma = torch.Tensor([1, 5, 10])

# d=1
# mu = torch.Tensor([10])
# sigma = torch.Tensor([5])
target_dist = dists.independent.Independent(dists.normal.Normal(mu, sigma), 1) # target distribution
start_dist = dists.independent.Independent(dists.normal.Normal(torch.zeros(d), torch.ones(d)), 1)
model = FlowModel(start_dist, *[Affine(d) for i in range(nb_flow)])

# Training parameters 
batch_size = 100
lr = 0.01
verbose = 1000
optim = torch.optim.Adam(model.parameters(), lr=lr)

now = datetime.now()
date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
writer = SummaryWriter(log_dir='runs/exercice1_' + date_time)

for epoch in range(epochs):
    # sample data from start dist
    xs = target_dist.sample(torch.Size([batch_size]))
    # compute transformation f(x) and prob p(f(x)) (ie from target dist in model) and log|det df(x)/dx|
    log_prob, _, Jf = model.invf(xs)
    optim.zero_grad()
    # compute loss and optimize
    negliklh = - (log_prob + Jf).mean()
    negliklh.backward()
    optim.step()
    # logging and printing info
    writer.add_scalar(f'Negative likelihodd', negliklh, epoch)
    if d==1:
        writer.add_scalar(f'sigma', torch.exp(model.flows[0].s), epoch)
        writer.add_scalar(f'mu', model.flows[0].t, epoch)
    else :
        writer.add_scalar(f'sigma', torch.exp(model.flows[0].s)[0], epoch)
        writer.add_scalar(f'mu', model.flows[0].t[0], epoch)
    if epoch%verbose==0:
        print(f'Epoch {epoch} : Negative Loglikelihood : {negliklh}')

print('estimated sigma', torch.exp(model.flows[0].s), 'true value', sigma)
print('estimated mu', model.flows[0].t, 'true value', mu)
