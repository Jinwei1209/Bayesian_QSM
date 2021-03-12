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
    flag_linear=1
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
    if flag_linear:
        diff = torch.abs(in_loss_RDFs_cplx - torch.ifft(cplx_mlpy(torch.fft(samples_cplx, 3), D_cplx), 3))
    else:
        samples_rdf = torch.ifft(cplx_mlpy(torch.fft(samples_cplx, 3), D_cplx), 3)
        diff_real = torch.cos(in_loss_RDFs_cplx[..., 0:1]) - torch.cos(samples_rdf[..., 0:1])
        diff_imag = torch.sin(in_loss_RDFs_cplx[..., 1:2]) - torch.sin(samples_rdf[..., 1:2])
        diff = torch.cat((diff_real, diff_imag), dim=-1)
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

def loss_QSMnet(outputs, QSMs, Masks, D):
    # l1 loss
    loss = lossL1()
    outputs = outputs[:, 0:1, ...]
    device = outputs.get_device()
    
    outputs_cplx = torch.zeros(*(outputs.size()+(2,))).to(device)
    outputs_cplx[..., 0] = outputs

    QSMs_cplx = torch.zeros(*(QSMs.size()+(2,))).to(device)
    QSMs_cplx[..., 0] = QSMs

    D = np.repeat(D[np.newaxis, np.newaxis, ..., np.newaxis], outputs.size()[0], axis=0)
    D_cplx = np.concatenate((D, np.zeros(D.shape)), axis=-1)
    D_cplx = torch.tensor(D_cplx, device=device).float()

    RDFs_outputs = torch.ifft(cplx_mlpy(torch.fft(outputs_cplx, 3), D_cplx), 3)
    RDFs_QSMs = torch.ifft(cplx_mlpy(torch.fft(QSMs_cplx, 3), D_cplx), 3)

    errl1 = loss(outputs*Masks, QSMs*Masks)
    errModel = loss(RDFs_outputs[..., 0]*Masks, RDFs_QSMs[..., 0]*Masks)
    errl1_grad = loss(abs(dxp(outputs))*Masks, abs(dxp(QSMs))*Masks) + loss(abs(dyp(outputs))*Masks, abs(dyp(QSMs))*Masks) + loss(abs(dzp(outputs))*Masks, abs(dzp(QSMs))*Masks)
    errModel_grad = loss(abs(dxp(RDFs_outputs[..., 0]))*Masks, abs(dxp(RDFs_QSMs[..., 0]))*Masks) + loss(abs(dyp(RDFs_outputs[..., 0]))*Masks, abs(dyp(RDFs_QSMs[..., 0]))*Masks) + loss(abs(dzp(RDFs_outputs[..., 0]))*Masks, abs(dzp(RDFs_QSMs[..., 0]))*Masks)
    errGrad = errl1_grad + errModel_grad
    return errl1 + errModel + 0.1*errGrad

def loss_FINE(outputs, in_loss_RDFs, fidelity_Ws, D, nonlinear=0, factor=3.1421):
    outputs = outputs[:, 0:1, ...]
    device = outputs.get_device()
    
    outputs_cplx = torch.zeros(*(outputs.size()+(2,))).to(device)
    outputs_cplx[..., 0] = outputs

    D = np.repeat(D[np.newaxis, np.newaxis, ..., np.newaxis], outputs.size()[0], axis=0)
    D_cplx = np.concatenate((D, np.zeros(D.shape)), axis=-1)
    D_cplx = torch.tensor(D_cplx, device=device).float()

    fidelity_Ws_cplx = torch.zeros(*(outputs.size()+(2,))).to(device)
    fidelity_Ws_cplx[..., 0] = fidelity_Ws

    RDFs_outputs = torch.ifft(cplx_mlpy(torch.fft(outputs_cplx, 3), D_cplx), 3)

    if nonlinear == 0:
        in_loss_RDFs_cplx = torch.zeros(*(outputs.size()+(2,))).to(device)
        in_loss_RDFs_cplx[..., 0] = in_loss_RDFs
        diff = torch.abs(in_loss_RDFs_cplx - RDFs_outputs)
        return torch.sum((fidelity_Ws_cplx*diff)**2)
    
    elif nonlinear == 1:
        RDF = RDFs_outputs[..., 0] * factor # radian
        exp_RDF = torch.zeros(*(outputs.size()+(2,))).to(device)
        exp_RDF[..., 0] = torch.cos(RDF)
        exp_RDF[..., 1] = torch.sin(RDF)
        RDF_measured = in_loss_RDFs * factor # radian
        exp_measured = torch.zeros(*(outputs.size()+(2,))).to(device)
        exp_measured[..., 0] = torch.cos(RDF_measured)
        exp_measured[..., 1] = torch.sin(RDF_measured)
        diff = cplx_mlpy(fidelity_Ws_cplx, exp_RDF - exp_measured)
        return  torch.sum(cplx_mlpy(diff, cplx_conj(diff)))


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


def cplx_conj(a):
    """
    conjugate of a complex number
    """
    device = a.get_device()
    out = torch.empty(a.shape).to(device)
    out[..., 0] = a[..., 0]
    out[..., 1] = -a[..., 1]
    return out


def lossL1():
    return nn.L1Loss()

def lossL2():
    return nn.MSELoss()

        