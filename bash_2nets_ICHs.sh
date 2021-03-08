#!/bin/bash
# IDs=(8 14 16)

IDs=(8 14)

for optm in {0..1}; do
	for id in "${IDs[@]}";do
			echo "ID: $id, optm: $optm"
			python main_FINE_resnet.py --optm=$optm --patientID=$id --alpha=0.5 --rho=30 --loader=1
			python main_FINE_resnet.py --optm=$optm --patientID=$id --alpha=1.0 --rho=60 --loader=1
			python main_FINE_resnet.py --optm=$optm --patientID=$id --alpha=0.0 --rho=00 --loader=1
	done
done
