#!/bin/bash
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=0             # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=00-02:59:00  # time (DD-HH:MM)
#SBATCH --account=def-guibault
#SBATCH --output=%N-%j.out

echo 'Start'
SOURCEDIR=~/scratch/fewShotnonperfectUpper4k14
# prepare virtualenv
source ~/ENV/bin/activate

#prepare data
cd $SLURM_TMPDIR
tar xf ~/scratch/fewShotnonperfectUpper4k14/archesMissingTeeth.tar 
cd archesMissingTeeth
echo "Staring MeshSegNet training!"

python $SOURCEDIR/step3_trainingSSKNN.py 

echo 'Finish'


