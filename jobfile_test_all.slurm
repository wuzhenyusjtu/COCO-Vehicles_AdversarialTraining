#!/bin/bash
#SBATCH -J visdrone_fpn_test
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -o visdrone_output_test_all.o
#SBATCH -e error_file_test.e
#SBATCH --gres=gpu:1
module purge
module load anaconda3
source activate py27t04


echo "soft nms, varying sigmas"
for sigma in $(seq 0.2 0.1 1.1)
	do
	echo "sigma is: $sigma"
	/home/ksuresh/.conda/envs/py27t04/bin/python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 15 --checkpoint 1272 --load_dir "./fpn_remove_6" --sig $sigma --cuda
done 
echo job done!