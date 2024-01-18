SEED=1
TRAINER=AutoGroup_EMA_EATA
GROUPS=3

# CUDA_VISIBLE_DEVICES=0 nohup python train.py --root ./data --seed ${SEED} --trainer ${TRAINER} --dataset-config-file configs/datasets/domainnet_clipart.yaml --config-file configs/trainers/PromptAlign/DG_PAlign_vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml --output-dir output/evaluation/${TRAINER}/logs --model-dir weights/maple/ori/seed1 --load-epoch 2 \
#  --tpt DATASET.NUM_SHOTS 16 TPT.N_VIEWS 64 TPT.BATCH_SIZE 1 TPT.LR 0.005 TPT.GROUPS ${GROUPS} TPT.TPT_LOSS False TPT.AUG_LOSS True TPT.DISTR_ALIGN False TPT.PLOGA_LOSS True >output/evaluation/${TRAINER}/groups${GROUPS}_seed${SEED}_reset${RESET_STEPS}.out

RESET_STEPS=2000
CUDA_VISIBLE_DEVICES=0 nohup python train.py --root ./data --seed ${SEED} --trainer ${TRAINER} --dataset-config-file configs/datasets/domainnet_clipart.yaml --config-file configs/trainers/PromptAlign/DG_PAlign_vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml --output-dir output/evaluation/${TRAINER}/logs --model-dir weights/maple/ori/seed1 --load-epoch 2 \
 --tpt DATASET.NUM_SHOTS 16 TPT.N_VIEWS 64 TPT.BATCH_SIZE 1 TPT.LR 0.005 TPT.GROUPS ${GROUPS} TPT.TPT_LOSS True TPT.AUG_LOSS False TPT.DISTR_ALIGN True TPT.PLOGA_LOSS False TPT.RESET_STEPS ${RESET_STEPS} >output/evaluation/${TRAINER}/groups${GROUPS}_seed${SEED}_reset${RESET_STEPS}.out
