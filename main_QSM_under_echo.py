import os
import time 
import numpy as np 
import torch
import torch.optim as optim
import random
import argparse

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
# from loader.COSMOS_data_loader import COSMOS_data_loader
from loader.QSM_data_loader_under_echo import QSM_data_loader_under_echo
from models.unet import Unet
from models.unetag import UnetAg
from utils.train import BayesianQSM_train
from utils.medi import *
from utils.loss import *
from utils.files import *

if __name__ == '__main__':

    # default parameters
    niter = 1000
    lr = 1e-3
    batch_size = 1
    voxel_size = (1, 1, 2)

    # typein parameters
    parser = argparse.ArgumentParser(description='Deep Learning QSM')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lambda_tv', type=float, default=1.0)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--weight_dir', type=str, default='weights_QSM_echo')
    opt = {**vars(parser.parse_args())}

    lambda_tv = opt['lambda_tv']
    flag_RDF_input = 1

    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_id'] 
    t0 = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rootDir = '/data/Jinwei/QSM_raw_CBIC/' + opt['weight_dir']

    # network
    unet3d = Unet(
        input_channels=1, 
        output_channels=1, 
        num_filters=[2**i for i in range(5, 10)],
        use_deconv=1,
        flag_rsa=0
    )
    print(unet3d)
    unet3d.to(device)

    if opt['train'] == 1:
        # optimizer
        optimizer = optim.Adam(unet3d.parameters(), lr = lr, betas=(0.5, 0.999))
        ms = [0.3, 0.5, 0.7, 0.9]
        ms = [np.floor(m * niter).astype(int) for m in ms]
        scheduler = MultiStepLR(optimizer, milestones = ms, gamma = 0.5)
        
        # logger
        logger = Logger('logs', rootDir)

        epoch = 0
        gen_iterations = 1
        display_iters = 5
        loss_fidelity_sum, loss_tv_sum = 0, 0
        Validation_loss = []
        loss_L1 = lossL1()

        # dataloader
        dataLoader_train = QSM_data_loader_under_echo(
            dataFolder = '/data/Jinwei/QSM_raw_CBIC/data_chao_echo_no_csf',
            split='train',
            flag_RDF_input=flag_RDF_input
        )
        trainLoader = data.DataLoader(dataLoader_train, batch_size=batch_size, shuffle=True, pin_memory=True)

        dataLoader_val = QSM_data_loader_under_echo(
            dataFolder = '/data/Jinwei/QSM_raw_CBIC/data_chao_echo_no_csf',
            split='val',
            flag_RDF_input=flag_RDF_input
        )
        valLoader = data.DataLoader(dataLoader_val, batch_size=batch_size, shuffle=False, pin_memory=True)

        # dataLoader_train = dataLoader_val
        # trainLoader = valLoader

        while epoch < niter:
            epoch += 1

            # training phase
            unet3d.train()
            for idx, (rdf_inputs, rdfs, masks, weights, wGs, D) in enumerate(trainLoader):
                if gen_iterations%display_iters == 0:
                    print('epochs: [%d/%d], batchs: [%d/%d], time: %ds'
                        % (epoch, niter, idx, dataLoader_train.num_subs//batch_size, time.time()-t0))
                    print('Fidelity loss: %f, TV loss: %f' % (loss_fidelity_sum/display_iters, loss_tv_sum/display_iters))
                    if epoch > 1:
                        print('Validation loss of last epoch: %f' % (Validation_loss[-1]))
                    loss_fidelity_sum, loss_tv_sum = 0, 0

                rdf_inputs = rdf_inputs.to(device, dtype=torch.float)
                rdfs = rdfs.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                wGs = wGs.to(device, dtype=torch.float)
                
                loss_fidelity, loss_tv = BayesianQSM_train(
                    model=unet3d,
                    input_RDFs=rdf_inputs,
                    in_loss_RDFs=rdfs,
                    QSMs=0,
                    Masks=masks,
                    fidelity_Ws=weights,
                    gradient_Ws=wGs,
                    D=np.asarray(D[0, ...]),
                    flag_COSMOS=0,
                    optimizer=optimizer,
                    sigma_sq=0,
                    Lambda_tv=lambda_tv,
                    voxel_size=voxel_size,
                    K=1,
                    flag_l1=2
                )

                loss_fidelity_sum += loss_fidelity
                loss_tv_sum += loss_tv
                gen_iterations += 1
            scheduler.step(epoch)

            # validation phase
            unet3d.eval()
            loss_total = 0
            idx = 0
            with torch.no_grad():
                for idx, (rdf_inputs, rdfs, masks, weights, wGs, D) in enumerate(valLoader):
                    idx += 1
                    rdf_inputs = rdf_inputs.to(device, dtype=torch.float)
                    rdfs = rdfs.to(device, dtype=torch.float)
                    masks = masks.to(device, dtype=torch.float)
                    weights = weights.to(device, dtype=torch.float)
                    wGs = wGs.to(device, dtype=torch.float)

                    outputs = unet3d(rdf_inputs)
                    # fidelity loss
                    loss_fidelity = loss_FINE(outputs, rdfs, weights, np.asarray(D[0, ...]))
                    # TV prior
                    grad = torch.zeros(*(outputs.size()+(3,))).to('cuda')
                    grad[..., 0] = dxp(outputs)/voxel_size[0]
                    grad[..., 1] = dyp(outputs)/voxel_size[1]
                    grad[..., 2] = dzp(outputs)/voxel_size[2]
                    loss_tv = lambda_tv*torch.sum(torch.abs(wGs*grad))/(2*outputs.size()[0])
                    loss = loss_fidelity + loss_tv
                    loss_total += loss

                print('\n Validation loss: %f \n' % (loss_total / idx))
                Validation_loss.append(loss_total / idx)

            logger.print_and_save('Epoch: [%d/%d], Loss in Validation: %f' 
            % (epoch, niter, Validation_loss[-1]))

            if Validation_loss[-1] == min(Validation_loss):
                torch.save(unet3d.state_dict(), rootDir+'/weight_lambda_tv={}.pt'.format(lambda_tv))
            torch.save(unet3d.state_dict(), rootDir+'/weight_last_lambda_tv={}.pt'.format(lambda_tv))

    elif opt['train'] == 0:
        # dataloader
        dataLoader = QSM_data_loader_under_echo(
            dataFolder = '/data/Jinwei/QSM_raw_CBIC/data_chao_echo_no_csf',
            split='val',
            flag_RDF_input=flag_RDF_input
        )
        Loader = data.DataLoader(dataLoader, batch_size=batch_size, shuffle=False, pin_memory=True)

        weights_dict = torch.load(rootDir+'/weight_last_lambda_tv={}.pt'.format(lambda_tv))
        unet3d.load_state_dict(weights_dict)
        unet3d.eval()
        QSMs = np.zeros((256, 206, 80, dataLoader.num_subs))

        with torch.no_grad():  # to solve memory exploration issue
            for idx, (rdf_inputs, rdfs, masks, weights, wGs, D) in enumerate(Loader):
                rdf_inputs = rdf_inputs.to(device, dtype=torch.float)
                rdfs = rdfs.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.float)
                weights = weights.to(device, dtype=torch.float)
                wGs = wGs.to(device, dtype=torch.float)
                outputs = unet3d(rdf_inputs) * masks
                QSMs[..., idx] = np.squeeze(outputs.cpu().detach())

        adict = {}
        adict['QSMs'] = QSMs
        sio.savemat('/data/Jinwei/QSM_raw_CBIC/tmp/QSMs_6.mat', adict)
