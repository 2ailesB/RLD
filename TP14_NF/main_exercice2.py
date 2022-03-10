import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from actnorm import actNorm
from conv1x1 import conv1x1
from couplinglayer import CouplingLayer
from utils import FlowModel
from torch.utils.data import DataLoader
import torch.distributions as dists


from sklearn import datasets

if __name__ == '__main__':

    # Data parameters
    torch.manual_seed(1111)
    n_samples = 10000
    batch_size = 100
    # ds, _ = datasets.make_circles(
    #     n_samples=n_samples, factor=0.5, noise=0.05, random_state=0)
    # train_dataloader = DataLoader(torch.from_numpy(ds), batch_size=batch_size, shuffle=True, num_workers=4)
    # nBatchTrain = len(train_dataloader)

    # Model parameters
    d = 2
    hdim_cl = 100
    nflow = 8
    start_dist = dists.independent.Independent(
        dists.normal.Normal(torch.zeros(d), torch.ones(d)), 1)

    # Model declaration
    convs = [conv1x1(indim=d, outdim=d) for i in range(nflow)]
    norms = [actNorm(d) for _ in range(nflow)]
    couplings = [CouplingLayer(indim=d, hdim=hdim_cl, invx=i % 2)
                 for i in range(nflow)]
    flows = []
    for cv, nm, cp in zip(convs, norms, couplings):
        flows += [nm, cv, cp]
    model = FlowModel(start_dist, *flows)

    # training parameters
    lr = 0.0001
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    epochs = 10000
    verbose = 500
    verbose_size = 1000
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
    logging_path = 'runs' + '/exercice2_' + date_time
    writer = SummaryWriter(log_dir=logging_path)

    for epoch in range(epochs):

        data, _ = datasets.make_moons(
            n_samples=batch_size, shuffle=True, noise=0.05, random_state=0)

        sample = torch.from_numpy(data).float()
        # forward pass
        logprob, zs, logdet = model.f(sample)
        # optimisation
        optim.zero_grad()
        negliklh = - (logprob + logdet).mean()
        negliklh.backward()
        optim.step()

        #log info
        writer.add_scalar(f'Negative likelihood', negliklh.item(), epoch)
        writer.add_scalar(f'Log(det|Jf|)', logdet.mean().item(), epoch)
        writer.add_scalar(f'logprob', logprob.mean().item(), epoch)
        if epoch % verbose == 0:
            print(f'Epoch {epoch} : Negative Loglikelihood : {negliklh}')
            with torch.no_grad():
                xs = start_dist.sample(torch.Size([batch_size]))
                _, batch_z, _ = model.invf(xs)  # batch_z is [1000, 2]
                fig = plt.figure()
                plt.scatter(batch_z[-1][:, 0], batch_z[-1][:, 1])
                writer.add_figure(f'output', fig, epoch)
                plt.clf()
                with torch.no_grad():
                    fig = plt.figure(figsize=[15, 45])
                    plt.subplot(nflow + 1, 3, 1)
                    plt.scatter(data[:, 0], data[:, 1])
                    for i in range(len(batch_z)):
                        plt.subplot(nflow + 1, 3, i+2)
                        plt.scatter(batch_z[i][:, 0], batch_z[i][:, 1])
                    writer.add_figure(f'layers', fig, epoch)
                    plt.clf()

    # final plot
    fig = plt.figure(figsize=[15, 45])
    prior_sample = start_dist.sample((verbose_size,))
    with torch.no_grad():
        _, output, _ = model.invf(prior_sample)

    plt.subplot(nflow + 1, 3, 1)
    plt.scatter(data[:, 0], data[:, 1])
    for i in range(len(output)):
        plt.subplot(nflow + 1, 3, i+2)
        plt.scatter(output[i][:, 0], output[i][:, 1])
    plt.savefig(logging_path + '/output.png')
    plt.clf()
