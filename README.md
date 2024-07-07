# Warning: This code is a WIP:

# Mistral Training

## Install Dependencies
To install the dependencies, please run the following command:
```
bash install.sh
```

## Dataset Preparation

1. Download the following datasets:

```
vietnamese_v1_reupload
english_v1
english_tiny-textbooks
english_code_v1
english_mini-peS2o
```

2. Combine them into a single dataset, shuffe it, and upload it to HF. The path to this dataset is: DATASET_PATH. Make sure you have enough space in your disk.

3. Update file ``prepare_data.sh`` with the path and run it to prepare the data.

Then, you need to change the path in file `src/llama_recipes/datasets/streaming_dataset` to your own absolute path.

## Training  

After preparing the dataset, you can start training the model. The following command will train the model with the default parameters. 
You should change the necessary parameters in the command, such as partition name, absolute path, etc.

```bash
torchrun --nnodes 1 --nproc_per_node 4 examples/finetuning.py \
        --enable_fsdp --fsdp_config.pure_bf16 \
        --model_name <THE_MODEL_PATH> \
        --batch_size_training=4 \
        --lr=5e-5 \
        --dist_checkpoint_root_folder <OUTPUT_PATH> \
        --dist_checkpoint_folder <CHECKPOINT_NAME> \
        --use_streaming_data \
        --streaming_dataset_path <YOUR_LOCAL_DATA_PATH> \
        --save_step=1000
```
