#!/bin/bash
#cd ../..

# custom config
DATA="./data"
TRAINER=PromptAlign

DATASET=$1
SEED=$2
# CUSTOM_NAME=$3
BATCH_SIZE=$3
CUDA_ID=$4
WEIGHTSPATH='weights/maple/ori'

CFG=DG_MapleDistrAlign_vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16
LOADEP=2

MODEL_DIR=${WEIGHTSPATH}/seed${SEED}

DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}

echo "Evaluating model"
echo "Runing the first phase job and save the output to ${DIR}"
# Evaluate on evaluation datasets
CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--tpt \
DATASET.NUM_SHOTS ${SHOTS} \
TPT.BATCH_SIZE ${BATCH_SIZE} \

# if [ -d "$DIR" ]; then
#     echo "Results are already available in ${DIR}. Skipping..."
# else
#     echo "Evaluating model"
#     echo "Runing the first phase job and save the output to ${DIR}"
#     # Evaluate on evaluation datasets
#     python train.py \
#     --root ${DATA} \
#     --seed ${SEED} \
#     --trainer ${TRAINER} \
#     --dataset-config-file configs/datasets/${DATASET}.yaml \
#     --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
#     --output-dir ${DIR} \
#     --model-dir ${MODEL_DIR} \
#     --load-epoch ${LOADEP} \
#     --tpt \
#     DATASET.NUM_SHOTS ${SHOTS} \

# fi