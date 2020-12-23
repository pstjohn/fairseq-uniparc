#!/bin/bash
#SBATCH --account=deepgreen
#SBATCH --time=2-00
#SBATCH --job-name=preprocess_fairseq
#SBATCH --nodes=21
#SBATCH --qos=high
#SBATCH --output=/scratch/pstjohn/fairseq.%j.out

source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh
conda activate /projects/deepgreen/pstjohn/envs/fairseq

for ((i = 0 ; i < 21 ; i++)); do
    iz=`printf "%02d" $i`
    srun -l -n 1 --nodes=1 fairseq-preprocess \
        --only-source \
        --srcdict /projects/deepgreen/pstjohn/dict.txt \
        --trainpref /projects/deepgreen/pstjohn/split_train/train_$iz.txt \
        --destdir /projects/deepgreen/pstjohn/split_bin/train_$iz/ \
        --workers 72 &
        
done

wait