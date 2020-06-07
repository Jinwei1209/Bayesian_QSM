#!/bin/bash
read -p "Enter GPU_ID: " id
read -p "Enter number of iters: " niter

for patientID in {1..7};
do
    python main_QSM_patient_all.py --gpu_id=$id --lambda_tv=20 --niter=$niter --flag_test=1 --flag_r_train=1 --patient_type='MS_new' --patient_ID=$patientID
done