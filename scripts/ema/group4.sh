CUDA_VISIBLE_DEVICES=2 nohup python train.py --root ./data --seed 1 --trainer Group_EMA_EATA --dataset-config-file configs/datasets/domainnet_clipart.yaml --config-file configs/trainers/PromptAlign/DG_PAlign_vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml --output-dir output/evaluation/Group_EMA/seed1 --model-dir weights/maple/ori/seed1 --load-epoch 2 \
 --tpt DATASET.NUM_SHOTS 16 TPT.N_VIEWS 64 TPT.BATCH_SIZE 1 TPT.LR 0.005 TPT.GROUPS 4 \ 
 TPT.TPT_LOSS False TPT.AUG_LOSS True TPT.DISTR_ALIGN False TPT.PLOGA_LOSS True >output/evaluation/Group_EMA_EATA/4.out

 CUDA_VISIBLE_DEVICES=2 nohup python train.py --root ./data --seed 1 --trainer Group_EMA_EATA --dataset-config-file configs/datasets/domainnet_clipart.yaml --config-file configs/trainers/PromptAlign/DG_PAlign_vit_b16_c2_ep5_batch4_2ctx_cross_datasets.yaml --output-dir output/evaluation/Group_EMA/seed1 --model-dir weights/maple/ori/seed1 --load-epoch 2 \
 --tpt DATASET.NUM_SHOTS 16 TPT.N_VIEWS 64 TPT.BATCH_SIZE 1 TPT.LR 0.005 TPT.GROUPS 4 TPT.RESET_STEPS 2000\ 
 TPT.TPT_LOSS False TPT.AUG_LOSS True TPT.DISTR_ALIGN False TPT.PLOGA_LOSS True >output/evaluation/Group_EMA_EATA/4_reset2000.out