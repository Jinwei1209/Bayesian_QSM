#!/bin/bash
read -p "Enter GPU_ID: " id
read -p "flag_VI: " vi

for i in {1..100}
do
	echo "Simulation: $i"
    python main_simulation_ich.py --idx $i --gpu_id $id --flag_test=1 --flag_VI=$vi
done
