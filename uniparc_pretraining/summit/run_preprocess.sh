##PSJ note -- this was actually done on eagle by sharding the dataset first and then running fairseq-preprocess on each of 21 shards.

#!/bin/bash

#BSUB -P BIE108
#BSUB -q batch
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -J preprocess_fairseq
#BSUB -o /ccs/home/pstjohn/fairseq_job_output/%J.out
#BSUB -e /ccs/home/pstjohn/fairseq_job_output/%J.err

module load ibm-wml-ce/1.7.1.a0-0
conda activate fairseq_1.7.1

PROJECTWORK=/gpfs/alpine/bie108/proj-shared
SRC_DIR=$PROJECTWORK/split_uniref100

jsrun -n1 fairseq-preprocess \
    --only-source \
    --srcdict dict.txt \
    --validpref $SRC_DIR/dev_uniref50.txt \
    --testpref $SRC_DIR/test_uniref50.txt \
    --destdir $PROJECTWORK/uniref_100_bin \
    --workers 60
