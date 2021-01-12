#!/bin/bash
declare -i val
for optm in {0..1}; do
	for alpha in `seq 0.0 0.1 1.0`;do
		for rho in `seq 10 10 100`;do
			echo "alpha: $alpha, rho: $rho, optm: $optm"
			python main_FINE_resnet.py --optm=$optm --patientID=14 --alpha=$alpha --rho=$rho
		done
	done
done
