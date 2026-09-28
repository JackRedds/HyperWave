RUN_TYPE="cs"
HW_OUTDIR=/projects/bcse/jredepenning/HyperWave-outdir/${RUN_TYPE}/
HW_OUTFILE=${HW_OUTDIR}/${RUN_TYPE}_reconstruction

source /projects/bcse/jredepenning/HyperWave/.venv/bin/activate

python /projects/bcse/jredepenning/HyperWave/scripts/${RUN_TYPE}_wavelet_reconstruction.py --device gpu \
    --proposal flowfisher --converge --check-every 200 --target-ess 3000 --thin 6 \
    --nsteps 24000 --burn 2000 --nwalkers 50 --ntemps 20 --nleaves-max 50 --seed 0 \
    --recon-plot ${HW_OUTDIR} --outfile ${HW_OUTFILE}