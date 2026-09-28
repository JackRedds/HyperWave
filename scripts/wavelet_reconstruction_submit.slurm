#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cs_reconstruction
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/jredepenning/hyperwave_outdir/%x.out
#SBATCH --error=/scratch/bcse/jredepenning/hyperwave_outdir/%x.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=164G
#SBATCH --mail-type=END
#SBATCH --mail-user=redep028@umn.edu

RUN_TYPE="cs"
HW_OUTDIR=/projects/bcse/jredepenning/HyperWave-outdir/${RUN_TYPE}/
HW_OUTFILE=${HW_OUTDIR}/${RUN_TYPE}_reconstruction

source /projects/bcse/jredepenning/HyperWave/.venv/bin/activate

python /projects/bcse/jredepenning/HyperWave/scripts/${RUN_TYPE}_wavelet_reconstruction.py --device gpu \
    --proposal flowfisher --converge --check-every 200 --target-ess 3000 --thin 6 \
    --nsteps 24000 --burn 2000 --nwalkers 50 --ntemps 20 --nleaves-max 50 --seed 0 \
    --recon-plot ${HW_OUTDIR} --outfile ${HW_OUTFILE}