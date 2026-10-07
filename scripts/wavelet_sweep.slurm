#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=wavelet_reconstruction
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/jredepenning/hyperwave_outdir/%x.out
#SBATCH --error=/scratch/bcse/jredepenning/hyperwave_outdir/%x.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=164G
#SBATCH --mail-type=END
#SBATCH --mail-user=redep028@umn.edu
#SBATCH --array=0-19

RUN_TYPE="cs"
HW_OUTDIR=/projects/bcse/jredepenning/HyperWave-outdir/${RUN_TYPE}/
HW_OUTFILE=${HW_OUTDIR}/${RUN_TYPE}_reconstruction

source /projects/bcse/jredepenning/HyperWave/.venv/bin/activate


python /projects/bcse/jredepenning/HyperWave/scripts/wavelet_param_sweep.py \
    --waveform all --sweep snr=10,20,30,40,60,80,100 --indices ${SLURM_ARRAY_TASK_ID} \
    --outdir /projects/bcse/jredepenning/HyperWave-outdir/snr_sweep --recon-plots \
    --device gpu --proposal flowfisher --converge --check-every 200 --target-ess 3000 --thin 6 \
    --nsteps 24000 --burn 2000 --nwalkers 50 --ntemps 20 --nleaves-max 50 --seed 0