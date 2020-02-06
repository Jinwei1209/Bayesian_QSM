import torch
import torch.nn as nn
import numpy as np
from utils.medi import *

from utils.data import *
from utils.files import *
from utils.test import *

def loss_KL(
    outputs,
    QSMs,
    flag_COSMOS,
    sigma_sq
):
    mean_Maps = outputs[:, 0:1, ...]
    var_Maps = outputs[:, 1:2, ...]
    if flag_COSMOS:
        return -1/2*torch.sum(var_Maps) \
               + torch.sum(torch.exp(var_Maps))/sigma_sq \
               + torch.sum((mean_Maps - QSMs)**2)/sigma_sq
    else:
        # return -1/2*torch.sum(var_Maps)
        return -1/2*torch.sum(torch.log(var_Maps))
        # return 0


def loss_Expectation(
    outputs,
    QSMs,
    in_loss_RDFs,
    fidelity_Ws,
    gradient_Ws,
    D,
    flag_COSMOS,
    Lambda_tv,
    voxel_size,
    K=1,
):

    device = outputs.get_device()

    # sampling
    mean_Maps = tile(outputs[:, 0:1, ...], 0, K)
    var_Maps = tile(outputs[:, 1:2, ...], 0, K)
    epsilon = torch.normal(mean=torch.zeros(*mean_Maps.size()), std=torch.ones(*var_Maps.size()))

    epsilon = epsilon.to(device, dtype=torch.float)
    # samples = mean_Maps + torch.exp(var_Maps/2)*epsilon
    samples = mean_Maps + torch.sqrt(var_Maps)*epsilon

    # samples, last dimension of size 2, representing the real and imaginary
    samples_cplx = torch.zeros(*(mean_Maps.size()+(2,))).to(device)
    samples_cplx[..., 0] = samples
    # # padding of samples_cplx
    # samples_cplx_padding = torch.zeros(samples_cplx.size()[0:4]+(samples_cplx.size(4)*2,)+(2,)).to(device)
    # samples_cplx_padding[..., samples_cplx.size(4)//2:samples_cplx.size(4)+samples_cplx.size(4)//2, :] = samples_cplx

    # rdf in fidelity term, last dimension of size 2
    in_loss_RDFs_cplx = torch.zeros(*(mean_Maps.size()+(2,))).to(device)
    in_loss_RDFs = tile(in_loss_RDFs, 0, K)
    in_loss_RDFs_cplx[..., 0] = in_loss_RDFs

    # adict = {}
    # adict['temp'] = np.asarray(samples_cplx_padding.cpu().detach())
    # sio.savemat('temp.mat', adict)

    # fidelity weights, last dimension of size 2
    fidelity_Ws_cplx = torch.zeros(*(mean_Maps.size()+(2,))).to(device)
    fidelity_Ws = tile(fidelity_Ws, 0, K)
    fidelity_Ws_cplx[..., 0] = fidelity_Ws

    # dipole kernel
    D = np.repeat(D[np.newaxis, np.newaxis, ..., np.newaxis], mean_Maps.size()[0], axis=0)
    D_cplx = np.concatenate((D, np.zeros(D.shape)), axis=-1)
    D_cplx = torch.tensor(D_cplx, device=device).float()

    # fidelity loss (version 1, only 2/3 centric is used)
    diff = torch.abs(in_loss_RDFs_cplx - torch.ifft(cplx_mlpy(torch.fft(samples_cplx, 3), D_cplx), 3))
    # loss = torch.sum((fidelity_Ws_cplx[..., diff.size(4)//6:diff.size(4)//6*5, :]*diff[..., diff.size(4)//6:diff.size(4)//6*5, :])**2)/(2*mean_Maps.size()[0])
    # fidelity loss (version 2, zero padding)
    # tmp = torch.ifft(cplx_mlpy(torch.fft(samples_cplx_padding, 3), D_cplx), 3)
    # diff = torch.abs(in_loss_RDFs_cplx - tmp[..., tmp.size(4)//2:tmp.size(4)+tmp.size(4)//2, :])
    loss = torch.sum((fidelity_Ws_cplx*diff)**2)/(2*mean_Maps.size()[0])
    
    if flag_COSMOS:
        return loss
    else:
        # TV prior
        gradient_Ws = tile(gradient_Ws, 0, K)
        grad = torch.zeros(*(mean_Maps.size()+(3,))).to(device)
        grad[..., 0] = dxp(samples)/voxel_size[0]
        grad[..., 1] = dyp(samples)/voxel_size[1]
        grad[..., 2] = dzp(samples)/voxel_size[2]
        loss_tv = Lambda_tv*torch.sum(torch.abs(gradient_Ws*grad))/(2*mean_Maps.size()[0])
        return loss, loss_tv

# tile torch tensor, each item in the dimension will be repeated K times before concatenate together
def tile(a, dim, n_tile):
    device = a.get_device()
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))

    # order_index = torch.tensor(order_index, device=device)
    order_index = order_index.to(device)
    return torch.index_select(a, dim, order_index)


def cplx_mlpy(a, b):
    """
    multiply two 'complex' tensors (with the last dim = 2, representing real and imaginary parts)
    """
    device = a.get_device()
    out = torch.empty(a.shape).to(device)
    out[..., 0] = a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1]
    out[..., 1] = a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
    return out


def lossL1():
    return nn.L1Loss()

def lossL2():
    return nn.MSELoss()

        