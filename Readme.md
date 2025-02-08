# Installation
```bash
pip install -r requirements.txt
```

# TACRED 5-shot Experiments

This section describes configurations for running TACRED 5-shot experiments. Adjust `CUDA_VISIBLE_DEVICES` according to your GPU setup.

## Batch Size 16 Configurations

```bash
# 1 augmentation
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 1 >> tacred-5shot-1nga-bz16.log

# 3 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 3 >> tacred-5shot-3nga-bz16.log

# 5 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 5 >> tacred-5shot-5nga-bz16.log

# 7 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 7 >> tacred-5shot-7nga-bz16.log

# 10 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 10 >> tacred-5shot-10nga-bz16.log
```

> Note: Configuration with 10 augmentations (num_gen_augment=10) is currently not available for batch size 16.

## Batch Size 32 Configurations

```bash
# 1 augmentation
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 >> tacred-5shot-1nga-bz32.log

# 3 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 3 >> tacred-5shot-3nga-bz32.log

# 5 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 5 >> tacred-5shot-5nga-bz32.log

# 7 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 7 >> tacred-5shot-7nga-bz32.log

# 10 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 10 >> tacred-5shot-10nga-bz32.log
```

> Note: Configuration with 10 augmentations (num_gen_augment=10) is currently not available for batch size 32.