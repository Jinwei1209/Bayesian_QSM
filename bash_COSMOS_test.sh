#!/bin/bash
read -p "Enter GPU_ID: " id

declare -i val
for i in 3
do
	echo "Test case: $i"
	if [ $i == 2 ]; then
		let val=7
	else
		let val=$i-1
	fi
	# python3 main_COSMOS_test.py --flag_rsa -1 --case_validation $val --case_test $i --gpu_id $id
	# python3 main_COSMOS_test.py --flag_rsa  0 --case_validation $val --case_test $i --gpu_id $id
	# python3 main_COSMOS_test.py --flag_rsa  1 --case_validation $val --case_test $i --gpu_id $id
	python3 main_COSMOS_test.py --flag_rsa  5 --case_validation $val --case_test $i --gpu_id $id --weight_dir weight_cv
	# python3 main_COSMOS_test.py --flag_rsa  3 --case_validation $val --case_test $i --gpu_id $id --weight_dir weight_cv2
done
