echo "Downloading data..."
python src/llama_recipes/datasets/prep_streaming_data.py \
    --dataset DATASET_PATH \
    --out_root YOUR_LOCAL_DATA_PATH \
    --concat_tokens 4096 \
    --tokenizer YOUR_TOKENIZER_PATH