#!/bin/bash

module load ibm-wml-ce/1.7.1.a0-0
conda activate fairseq_1.7.1

# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4

NODES=1
TOTAL_UPDATES=54246      # Total number of training steps
WARMUP_UPDATES=50        # Warmup the learning rate over this many updates
PEAK_LR=1e-05            # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024   # Max sequence length
MAX_POSITIONS=1024       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=4          # Number of sequences per batch (batch size)
UPDATE_FREQ=2            # Increase the batch size 2x

DATA_DIR=/gpfs/alpine/bie108/proj-shared/swissprot_go_annotation/fairseq_swissprot
ROBERTA_PATH=$MEMBERWORK/bie108/fairseq-uniparc/roberta_base_checkpoint/roberta.base_with_go_bias.pt

fairseq-train \
    --fp16 $DATA_DIR \
    --user-dir $HOME/fairseq-uniparc/go_annotation/ \
    --restore-file $ROBERTA_PATH \
    --classification-head-name='go_prediction' \
    --task sentence_labeling --criterion go_prediction --regression-target --num-classes 32012 \
    --arch roberta_base --max-positions $TOKENS_PER_SAMPLE --shorten-method='random_crop' \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --validate-interval-updates 500 \
    --save-interval-updates 500 \
    --batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 10 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --find-unused-parameters