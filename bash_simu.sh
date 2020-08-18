#!/bin/bash
read -p "Enter GPU_ID: " id
read -p "Enter SNR: " snr

for i in {1..100}
do
	echo "Simulation: $i"
    python main_simulation_test.py --idx $i --gpu_id $id --snr $snr
done
