import os
import time 
import numpy as np 
import torch
import random
from torch.utils import data
from loader.QSM_data_loader2 import QSM_data_loader2

if __name__ == '__main__':

    time0 = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

    dataLoader = QSM_data_loader2()
    trainLoader = data.DataLoader(dataLoader, batch_size=1, shuffle=False)

    for idx, (input_RDFs, in_loss_RDFs, QSMs, Masks, \
        fidelity_Ws, gradient_Ws, flag_COSMOS) in enumerate(trainLoader):
        print(flag_COSMOS)

    print('Total time is {}'.format(time.time() - time0))