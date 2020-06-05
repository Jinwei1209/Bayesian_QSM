import numpy as np
import torch
import torch.optim as optim

from torch.autograd import Variable
from utils.loss import *
from utils.medi import *


def BayesianQSM_train(
    model,
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
    outputs = model(input_RDFs)

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
            return loss_kl.item(), loss_tv.item(), loss_expectation.item()

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


def utfi_train(
    model,
    optimizer,
    ifreqs,
    masks,
    data_weights,
    wGs,
    D,
    D_smv,
    lambda_pdf,
    lambda_tv,
    voxel_size,
    flag_train=1
):
    if flag_train:
        optimizer.zero_grad()
    loss_l1 = lossL1()
    loss_l2 = lossL2()
    outputs = model(ifreqs)
    device = outputs.get_device()

    # convert to cplx tensors
    chi_b, chi_l = outputs[:, 0:1, ...], outputs[:, 1:2, ...]
    chi_b_cplx = torch.zeros(*(chi_b.size()+(2,))).to(device)
    chi_b_cplx[..., 0] = chi_b
    chi_l_cplx = torch.zeros(*(chi_l.size()+(2,))).to(device)
    chi_l_cplx[..., 0] = chi_l

    ifreqs_cplx = torch.zeros(*(ifreqs.size()+(2,))).to(device)
    ifreqs_cplx[..., 0] = ifreqs

    masks_bg = torch.zeros(*(masks.size()+(2,))).to(device)
    masks_bg[..., 0] = 1 - masks
    masks_lc = torch.zeros(*(masks.size()+(2,))).to(device)
    masks_lc[..., 0] = masks

    fidelity_Ws_cplx = torch.zeros(*(data_weights.size()+(2,))).to(device)
    fidelity_Ws_cplx[..., 0] = data_weights

    D = np.repeat(D[np.newaxis, np.newaxis, ..., np.newaxis], outputs.size()[0], axis=0)
    D_cplx = np.concatenate((D, np.zeros(D.shape)), axis=-1)
    D_cplx = torch.tensor(D_cplx, device=device).float()

    D_smv = np.repeat(D_smv[np.newaxis, np.newaxis, ..., np.newaxis], outputs.size()[0], axis=0)
    D_smv_cplx = np.concatenate((D_smv, np.zeros(D_smv.shape)), axis=-1)
    D_smv_cplx = torch.tensor(D_smv_cplx, device=device).float()

    # loss of PDF
    chi_b_cplx = cplx_mlpy(chi_b_cplx, masks_bg)
    f_chi_b = torch.ifft(cplx_mlpy(torch.fft(chi_b_cplx, 3), D_cplx), 3)
    data_term_b = cplx_mlpy(fidelity_Ws_cplx, f_chi_b - ifreqs_cplx)
    loss_PDF = lambda_pdf * torch.mean(data_term_b[..., 0]**2)

    # backgroud field removal
    RDF_cplx = cplx_mlpy(ifreqs_cplx - f_chi_b, masks_lc)

    # fidelity loss of MEDI, no smv, linear
    # chi_l_cplx = cplx_mlpy(chi_l_cplx, masks_lc)
    f_chi_l = torch.ifft(cplx_mlpy(torch.fft(chi_l_cplx, 3), D_cplx), 3)
    data_term_l = cplx_mlpy(fidelity_Ws_cplx, f_chi_l - RDF_cplx)
    loss_fidelity = torch.mean(data_term_l[..., 0]**2)

    # # fidelity loss of MEDI, no smv, nonlinear
    # f_chi_l = torch.ifft(cplx_mlpy(torch.fft(chi_l_cplx, 3), D_cplx), 3)
    # expi_f_chi_l = torch.cat((torch.cos(f_chi_l[..., 0:1]), torch.sin(f_chi_l[..., 0:1])), dim=-1)
    # expi_RDF_cplx = torch.cat((torch.cos(RDF_cplx[..., 0:1]), torch.sin(RDF_cplx[..., 0:1])), dim=-1)
    # data_term_l = cplx_mlpy(fidelity_Ws_cplx, expi_f_chi_l - expi_RDF_cplx)
    # loss_fidelity = torch.mean(data_term_l[..., 0]**2)

    # fidelity loss of MEDI, smv, nonlinear
    # TO DO

    # TV loss
    grad = torch.zeros(*(chi_l.size()+(3,))).to(device)
    grad[..., 0] = dxp(chi_l)/voxel_size[0]
    grad[..., 1] = dyp(chi_l)/voxel_size[1]
    grad[..., 2] = dzp(chi_l)/voxel_size[2]
    loss_tv = lambda_tv*torch.mean(torch.abs(wGs*grad))

    # Total loss
    loss_total = loss_PDF + loss_fidelity + loss_tv

    if flag_train:
        # Back-propogation
        loss_total.backward()
        optimizer.step()
    return loss_PDF.item(), loss_fidelity.item(), loss_tv.item()

def vae_train(model, optimizer, x, mask):
    optimizer.zero_grad()
    x_mu, x_var, z_mu, z_logvar = model(x)
    x_factor = torch.prod(torch.tensor(x.size()))
    z_factor = torch.prod(torch.tensor(z_mu.size()))
    print(x_factor, z_factor)
    # recon_loss = 0.5 * torch.sum((x_mu*mask - x*mask)**2 / (x_var + 1e-7) + torch.log(x_var)*mask) / x_factor
    recon_loss = 0.5 * torch.sum((x_mu - x)**2 / (x_var + 1e-7) + torch.log(x_var)) / x_factor
    # recon_loss = torch.sum((x_mu - x)**2) / x_factor
    kl_loss = -0.5*torch.sum(1 + z_logvar - z_mu**2 - torch.exp(z_logvar)) / z_factor
    total_loss = recon_loss + kl_loss * 0.1
    total_loss.backward()
    optimizer.step()
    return recon_loss.item(), kl_loss.item()



