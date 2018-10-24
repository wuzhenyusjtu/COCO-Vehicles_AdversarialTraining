#!/bin/bash
#SBATCH -J visdrone_fpn_test
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -o visdrone_output_test.o
#SBATCH -e error_file_test.e
#SBATCH --gres=gpu:1
module purge
module load anaconda3
#######################export LD_LIBRARY_PATH=/export/home/dyh/anaconda3/envs/py27t04/lib:$LD_LIBRARY_PATH
source activate py27t04


/home/ksuresh/.conda/envs/py27t04/bin/python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 12 --checkpoint 461 --cuda 
echo job done!