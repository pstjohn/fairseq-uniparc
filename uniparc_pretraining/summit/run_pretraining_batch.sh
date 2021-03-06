#!/bin/bash

#BSUB -P BIE108
#BSUB -q batch
#BSUB -W 12:00
#BSUB -nnodes 110
#BSUB -J 121320_roberta_base_batch
#BSUB -o /ccs/home/pstjohn/fairseq_job_output/%J.out
#BSUB -e /ccs/home/pstjohn/fairseq_job_output/%J.err
#BSUB -alloc_flags NVME
#BSUB -B

nnodes=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)

module load ibm-wml-ce/1.7.1.a0-0
conda activate fairseq_1.7.1

# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
TOTAL_UPDATES=500000     # Total number of training steps
WARMUP_UPDATES=24000     # Warmup the learning rate over this many updates
PEAK_LR=0.0007           # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024   # Max sequence length
MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=4          # Number of sequences per batch (batch size)
UPDATE_FREQ=3            # Increase the batch size 16x

# Overall batch size is nodes * 6 * max_sentences * update_freq
# For a batch size ~ 8k, nodes * update_freq should be ~333

SAVE_DIR=$MEMBERWORK/bie108/fairseq-uniparc/$LSB_JOBNAME
DATA_DIR=/gpfs/alpine/bie108/proj-shared/split_bin/

jsrun -n ${nnodes} -a 1 -c 42 -r 1 cp -r ${DATA_DIR} /mnt/bb/${USER}/

DATA_LIST=`ls -1 ~/project_work/split_bin | xargs echo | sed 's/ /:\/mnt\/bb\/pstjohn\/split_bin\//g' | sed 's/^/\/mnt\/bb\/pstjohn\/split_bin\//'`


jsrun -n ${nnodes} -g 6 -c 42 -r1 -a1 -b none \
    fairseq-train --distributed-port 23456 \
    --fp16 $DATA_LIST \
    --task masked_lm --criterion masked_lm \
    --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE --shorten-method='random_crop' \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ --save-dir $SAVE_DIR --save-interval 1 \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 1 --reset-dataloader
