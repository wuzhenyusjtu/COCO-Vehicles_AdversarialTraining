#!/bin/bash
#SBATCH -J visdrone_fpn_image_cut
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -o visdrone_output_image_cut_2.o
#SBATCH -e error_file_2.e
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu2
module purge
module load anaconda3
source activate py27t04
echo "starting job...."


/home/ksuresh/.conda/envs/py27t04/bin/python trainval_net.py --dataset pascal_voc --net res101 --bs 12 --cuda --mGPU
echo job done!