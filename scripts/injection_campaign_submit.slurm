#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sg_campaign
#SBATCH --account=bcse-delta-gpu
#SBATCH --output=/scratch/bcse/jredepenning/hyperwave_outdir/%x_%a.out
#SBATCH --error=/scratch/bcse/jredepenning/hyperwave_outdir/%x_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=164G
#SBATCH --array=0-19
#SBATCH --mail-type=END
#SBATCH --mail-user=redep028@umn.edu

# One injection per array task; N_INJ must cover the --array range. Each task
# re-aggregates campaign_summary.json from the inj_*.npz present when it ends;
# run with --summarize-only afterwards for the final table.
RUN_TYPE="sg"
N_INJ=20
HW_OUTDIR=/projects/bcse/jredepenning/HyperWave-outdir/${RUN_TYPE}_campaign/

source /projects/bcse/jredepenning/HyperWave/.venv/bin/activate

python /projects/bcse/jredepenning/HyperWave/scripts/wavelet_injection_campaign.py \
    --waveform ${RUN_TYPE} --n-injections ${N_INJ} --indices ${SLURM_ARRAY_TASK_ID} \
    --snr-range 10 40 --device gpu \
    --proposal flowfisher --converge --check-every 200 --target-ess 3000 --thin 6 \
    --nsteps 24000 --burn 2000 --nwalkers 50 --ntemps 20 --nleaves-max 50 --seed 0 \
    --recon-plots --outdir ${HW_OUTDIR}
