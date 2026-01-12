#!/usr/bin/env bash

python main.py     --model vit_base_patch16_dinov3.lvd1689m     --data-set IMNET100     --data-path /home/cpalaskas/data/imagenet100     --input-size 224     --output_dir ./prob_analysis     --prob_analysis    --batch-size 256 --finetune --color-jitter 0.0  --aa ""  --reprob 0.0  --mixup 0.0  --cutmix 0.0  --smoothing 0.0  --no-repeated-aug --output_dir plots_base_dinov3