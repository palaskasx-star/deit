#!/bin/bash

# Activate your environment
source /home/cpalaskas/virtual_envs/CHAD/bin/activate 

# The root folder where all your models are organized
BASE_DIR="./../models_report/organized_models"

# Define fixed hyperparameters for easy modification and folder naming
BATCH_SIZE=128
LR="5e-5"
LAYER_DECAY=1

# Array of all datasets you want to evaluate
DATASETS=("CIFAR" "FLOWERS" "PETS" "AIRCRAFT" "CARS")

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

    # 4. Iterate through each dataset and run transfer learning
    for DATASET in "${DATASETS[@]}"; do
        
        # Determine epochs and data path based on dataset size
        # Fine-grained, smaller datasets typically require more epochs to converge
        case "$DATASET" in
            "CIFAR")
                EPOCHS=50  # ~50,000 train images
                DATA_PATH="./../data/cifar100"
                ;;
            "FLOWERS")
                EPOCHS=100 # ~1,020 train images
                DATA_PATH="./../data/flowers"
                ;;
            "PETS")
                EPOCHS=100 # ~3,680 train images
                DATA_PATH="./../data/pets"
                ;;
            "AIRCRAFT")
                EPOCHS=100 # ~6,667 train images
                DATA_PATH="./../data/aircraft"
                ;;
            "CARS")
                EPOCHS=100 # ~8,144 train images
                DATA_PATH="./../data/cars"
                ;;
            *)
                echo "⚠️  Unknown dataset '$DATASET'. Skipping..."
                continue
                ;;
        esac

        # 5. Define the nested output directory inside this specific setup's folder
        # Format: dataset/epochs/batch_size/learning_rate/layer_decay
        out_dir="$setup_dir/transfer_results/$DATASET/$EPOCHS/$BATCH_SIZE/$LR/$LAYER_DECAY"
        mkdir -p "$out_dir"

        echo "===================================================================="
        echo "🚀 Starting Transfer Learning for: $arch | $method | $size"
        echo "📊 Dataset: $DATASET | Epochs: $EPOCHS"
        echo "📦 Model Arg: $model_arg"
        echo "🎯 Finetuning: $ckpt"
        echo "📂 Output Dir: $out_dir"
        echo "===================================================================="

        # 6. Run the training command
        python main.py \
            --model "$model_arg" \
            --finetune "$ckpt" \
            --data-path "$DATA_PATH" \
            --data-set "$DATASET" \
            --output_dir "$out_dir" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            --layer-decay "$LAYER_DECAY" \
            --epochs "$EPOCHS" \
            --unscale-lr \
            --drop-path 0.1

        echo "✅ Finished: $arch/$method/$size on $DATASET"
        echo ""

    done # End dataset loop
done # End model loop

echo "🎉 All Dinov3 transfer learning setups have completed across all datasets!"
