import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import argparse

from torch.utils import data
from loader.Patient_data_loader import Patient_data_loader
from loader.Patient_data_loader_all import Patient_data_loader_all
from models.unet import Unet
from models.resBlock import ResBlock
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

'''
    FINE of resnet on top of pre-trained unet3d and resnet
'''
if __name__ == '__main__':

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0, 1')
    parser.add_argument('--patient_type', type=str, default='ICH')  # or MS_old, MS_new
    parser.add_argument('--patientID', type=int, default=8)
    opt = {**vars(parser.parse_args())}

    # python main_QSM_patient.py --flag_r_train=1 --patientID=8 (or 16)

    patient_type = opt['patient_type']
    patientID = opt['patientID']

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/Bayesian_QSM'

    # dataloader
    dataLoader_train = Patient_data_loader(
        patientType=patient_type, 
        patientID=patientID,
        flag_input=1
    )

    # parameters
    niter = 100
    lr = 3e-4
    batch_size = 1
    B0_dir = (0, 0, 1)

    voxel_size = dataLoader_train.voxel_size
    volume_size = dataLoader_train.volume_size
    S = SMV_kernel(volume_size, voxel_size, radius=5)
    D = dipole_kernel(volume_size, voxel_size, B0_dir)
    D = np.real(S * D)

    trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True)

    # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=1, 
        num_filters=[2**i for i in range(5, 10)],  # or range(3, 8)
        use_deconv=1,
        flag_rsa=0
    )
    unet3d.to(device0)
    weights_dict = torch.load(rootDir+'/weight_2nets/unet3d_fine.pt')
    # weights_dict = torch.load(rootDir+'/weight_2nets2/linear_factor=1_validation=6_test=7_unet3d.pt')
    unet3d.load_state_dict(weights_dict)

    resnet = ResBlock(
        input_dim=2, 
        filter_dim=32,
        output_dim=1
    )
    resnet.to(device1)
    weights_dict = torch.load(rootDir+'/weight_2nets/resnet_fine.pt')
    # weights_dict = torch.load(rootDir+'/weight_2nets2/linear_factor=1_validation=6_test=7_resnet.pt')
    resnet.load_state_dict(weights_dict)

    # optimizer
    optimizer = optim.Adam(resnet.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 0
    loss_iters = np.zeros(niter)
    while epoch < 2:
        epoch += 1

        # training phase
        for idx, (rdf_inputs, rdfs, masks, weights, wGs, D_) in enumerate(trainLoader):

            if epoch == 1:
                unet3d.eval(), resnet.eval()
                rdf_inputs = rdf_inputs.to(device0, dtype=torch.float)
                qsm_inputs = unet3d(rdf_inputs).cpu().detach()
                QSMnet = np.squeeze(np.asarray(qsm_inputs))

            else:
                # to GPU device
                rdf_inputs = rdf_inputs.to(device1, dtype=torch.float)
                qsm_inputs1 = qsm_inputs.to(device1, dtype=torch.float)
                inputs_cat = torch.cat((rdf_inputs, qsm_inputs1), dim=1)
                rdfs = rdfs.to(device1, dtype=torch.float)
                masks = masks.to(device1, dtype=torch.float)
                weights = weights.to(device1, dtype=torch.float)
                wGs = wGs.to(device1, dtype=torch.float)

                # save initial QSM
                qsm_outputs = resnet(inputs_cat).cpu().detach()
                QSMnet = np.squeeze(np.asarray(qsm_outputs))
                print('Saving initial results')
                adict = {}
                adict['QSMnet'] = QSMnet
                sio.savemat(rootDir+'/QSMnet.mat', adict)

                # to complex array
                D = np.repeat(D[np.newaxis, np.newaxis, ..., np.newaxis], qsm_outputs.size()[0], axis=0)
                D_cplx = np.concatenate((D, np.zeros(D.shape)), axis=-1)
                D_cplx = torch.tensor(D_cplx, device=device1).float()

                in_loss_RDFs_cplx = torch.zeros(*(qsm_outputs.size()+(2,))).to(device1)
                in_loss_RDFs_cplx[..., 0] = rdfs

                fidelity_Ws_cplx = torch.zeros(*(qsm_outputs.size()+(2,))).to(device1)
                fidelity_Ws_cplx[..., 0] = weights
    
    epoch = 0
    t0 = time.time()
    while epoch < niter:
        epoch += 1
        optimizer.zero_grad()
        # forward
        outputs = resnet(inputs_cat)
        # to compelx array
        outputs = outputs[:, 0:1, ...]
        outputs_cplx = torch.zeros(*(outputs.size()+(2,))).to(device1)
        outputs_cplx[..., 0] = outputs
        # fidelity loss
        RDFs_outputs = torch.ifft(cplx_mlpy(torch.fft(outputs_cplx, 3), D_cplx), 3)
        diff = torch.abs(in_loss_RDFs_cplx - RDFs_outputs)
        loss = torch.sum((fidelity_Ws_cplx*diff)**2)
        # backward
        loss.backward()
        optimizer.step()

        print('epochs: [%d/%d], time: %ds, Fidelity loss: %f' % (epoch, niter, time.time()-t0, loss.item()))

    FINE = resnet(inputs_cat)[:, 0, ...]
    FINE = np.squeeze(np.asarray(FINE.cpu().detach()))

    adict = {}
    adict['FINE'] = FINE
    sio.savemat(rootDir+'/FINE.mat', adict)
