```bash
pip install -r requirements.txt
```

## Tacred 5shot
Adjust CUDA_VISIBLE_DEVICES to adapt your device
Need to run following configs

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 1 >> tacred-5shot-1nga-bz16.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 3 >> tacred-5shot-3nga-bz16.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 5 >> tacred-5shot-5nga-bz16.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 7 >> tacred-5shot-7nga-bz16.log
```

!! This config is not available at the moment !!
```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 16 --num_gen_augment 10 >> tacred-5shot-10nga-bz16.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 1 >> tacred-5shot-1nga-bz32.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 3 >> tacred-5shot-3nga-bz32.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 5 >> tacred-5shot-5nga-bz32.log
```

```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 7 >> tacred-5shot-7nga-bz32.log
```

!! This config is not available at the moment !!
```bash
CUDA_VISIBLE_DEVICES=0 python train_multi_k_tacred.py --task_name Tacred --num_k 5 --num_gen 5 --batch_size 32 --num_gen_augment 10 >> tacred-5shot-10nga-bz32.log
```