import numpy as np
import torch
import torch.optim as optim

from torch.autograd import Variable
from utils.loss import *


def BayesianQSM_train(
    unet3d,
    input_RDFs,
    in_loss_RDFs,
    QSMs,
    Masks,
    fidelity_Ws,
    gradient_Ws,
    D,
    flag_COSMOS,
    optimizer,
    sigma_sq,
    Lambda_tv,
    voxel_size,
    flag_l1=0,
    K=1
):

    optimizer.zero_grad()
    outputs = unet3d(input_RDFs)

    if not flag_l1:

        loss_kl = loss_KL(outputs, QSMs, flag_COSMOS, sigma_sq)

        if flag_COSMOS:
            loss_expectation = loss_Expectation(outputs, QSMs, in_loss_RDFs, fidelity_Ws, gradient_Ws, \
                                                D, flag_COSMOS, Lambda_tv, voxel_size, K)
            loss_total = loss_kl + loss_expectation
        else:
            loss_expectation, loss_tv = loss_Expectation(outputs, QSMs, in_loss_RDFs, fidelity_Ws, gradient_Ws, \
                                                        D, flag_COSMOS, Lambda_tv, voxel_size, K)
            loss_total = loss_kl + loss_expectation + loss_tv

        loss_total.backward()
        optimizer.step()

        if flag_COSMOS:
            return loss_kl.item(), loss_expectation.item()
        else:
            return (loss_kl+loss_tv).item(), loss_expectation.item()

    else:
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

        err = errl1 + errModel + 0.1*errGrad
        err.backward()
        optimizer.step()

        return err.item()