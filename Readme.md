# Installation
```bash
pip install -r requirements.txt
```

# TACRED 5-shot Experiments

This section describes configurations for running TACRED 5-shot experiments. Adjust `CUDA_VISIBLE_DEVICES` according to your GPU setup.

## Batch Size 32 Configurations

```bash
# 1 augmentation
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 2.0 --w2 2.0 --w3 0.5 >> tacred-5shot-1nga-bz32-202005.log

# 3 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 3 --w1 2.0 --w2 2.0 --w3 0.5 >> tacred-5shot-3nga-bz32-202005.log

# 5 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 5 --w1 2.0 --w2 2.0 --w3 0.5 >> tacred-5shot-5nga-bz32-202005.log

# 7 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 7 --w1 2.0 --w2 2.0 --w3 0.5 >> tacred-5shot-7nga-bz32-202005.log

# 10 augmentations
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 10 --w1 2.0 --w2 2.0 --w3 0.5 >> tacred-5shot-10nga-bz32-202005.log
```


## Tuning loss weights bz32 1 augmentation

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 2.0 --w2 2.0 --w3 2.0 >> tacred-5shot-1nga-bz32-202020.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 2.0 --w2 2.0 --w3 0.5 >> tacred-5shot-1nga-bz32-202005.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 2.0 --w2 0.5 --w3 2.0 >> tacred-5shot-1nga-bz32-200520.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 0.5 --w2 2.0 --w3 2.0 >> tacred-5shot-1nga-bz32-052020.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 2.0 --w2 0.5 --w3 0.5 >> tacred-5shot-1nga-bz32-200505.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 0.5 --w2 2.0 --w3 0.5 >> tacred-5shot-1nga-bz32-052005.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 0.5 --w2 0.5 --w3 2.0 >> tacred-5shot-1nga-bz32-050520.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 0.5 --w2 0.5 --w3 0.5 >> tacred-5shot-1nga-bz32-050505.log
```


# FEWREL 5-shot Experiments

```bash
# 1 augmentation
CUDA_VISIBLE_DEVICES=0 python train.py --task_name FewRel --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 --w1 2.0 --w2 2.0 --w3 0.5 >> fewrel-5shot-1nga-bz32-202005.log

# 3 augmentations
CUDA_VISIBLE_DEVICES=0 python train.py --task_name FewRel --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 3 --w1 2.0 --w2 2.0 --w3 0.5 >> fewrel-5shot-3nga-bz32-202005.log

# 5 augmentations
CUDA_VISIBLE_DEVICES=0 python train.py --task_name FewRel --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 5 --w1 2.0 --w2 2.0 --w3 0.5 >> fewrel-5shot-5nga-bz32-202005.log

# 7 augmentations
CUDA_VISIBLE_DEVICES=0 python train.py --task_name FewRel --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 7 --w1 2.0 --w2 2.0 --w3 0.5 >> fewrel-5shot-7nga-bz32-202005.log

# 10 augmentations
CUDA_VISIBLE_DEVICES=0 python train.py --task_name FewRel --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 10 --w1 2.0 --w2 2.0 --w3 0.5 >> fewrel-5shot-10nga-bz32-202005.log
```


# Baselines

## CPL-MI

### Setup

1. Navigate to the CPL-MI directory:
```bash
cd baselines/CPL-MI
```

2. Create environment file:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```


#### TACRED Dataset
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --task_name Tacred \
    --num_k 5 \
    --num_gen 5 \
    >> logs/tacred-5shot-bz32-k5-g5-mi.log
```

#### FewRel Dataset
```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
    --task_name FewRel \
    --num_k 5 \
    --num_gen 2 \
    >> logs/fewrel-5shot-bz32-k5-g2-mi.log
```
