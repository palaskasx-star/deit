#!/bin/bash

# Activate your environment
source /home/cpalaskas/virtual_envs/CHAD/bin/activate 

# The root folder where all your models are organized
BASE_DIR="./../models_report/organized_models"

# Iterate ONLY over Dinov3 directories: BASE_DIR / Dinov3 / Method / Size
for setup_dir in "$BASE_DIR"/Dinov3/*/*; do
    
    # Ensure it's actually a directory
    if [ ! -d "$setup_dir" ]; then
        continue
    fi

    # 1. Extract Architecture and Size from the folder path
    size=$(basename "$setup_dir")
    method=$(basename $(dirname "$setup_dir"))
    arch=$(basename $(dirname $(dirname "$setup_dir")))

    # 2. Find the 300-epoch checkpoint inside this directory
    ckpt=$(find "$setup_dir" -maxdepth 1 -name "*300.pth*" | head -n 1)

    if [ -z "$ckpt" ]; then
        echo "⚠️  No 300-epoch checkpoint found in $arch/$method/$size. Skipping..."
        continue
    fi

    # 3. Determine the correct --model argument based on Arch and Size
    model_arg=""
    size_lower=$(echo "$size" | tr '[:upper:]' '[:lower:]')

    # We already filtered for Dinov3 in the loop, but keeping this for safety
    if [[ "$arch" == "Dinov3" ]]; then
        model_arg="vit_${size_lower}_patch16_dinov3"
    else
        echo "⚠️  Unknown architecture '$arch'. Cannot map to a model name. Skipping..."
        continue
    fi

    # 4. Define the output directory inside this specific setup's folder
    out_dir="$setup_dir/transfer_cifar100"
    mkdir -p "$out_dir"

    echo "===================================================================="
    echo "🚀 Starting Transfer Learning for: $arch | $method | $size"
    echo "📦 Model Arg: $model_arg"
    echo "🎯 Finetuning: $ckpt"
    echo "📂 Output Dir: $out_dir"
    echo "===================================================================="

    # 5. Run the training command
    python main.py \
        --model "$model_arg" \
        --finetune "$ckpt" \
        --data-path ./../data/cifar100 \
        --data-set CIFAR \
        --output_dir "$out_dir" \
        --batch-size 128 \
        --lr 5e-5 \
        --layer-decay 1 \
        --epochs 50 \
        --unscale-lr \
        --drop-path 0.1

    echo "✅ Finished: $arch/$method/$size"
    echo ""

done

echo "🎉 All Dinov3 transfer learning setups have completed!"
