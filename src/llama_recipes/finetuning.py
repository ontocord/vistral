# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os, json
from pkg_resources import packaging

import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.distributed as dist

from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer
)
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llama_recipes.modeling.modeling_fast_llama import LlamaForCausalLM
from llama_recipes.modeling.modeling_fast_llama import LlamaDecoderLayer

from llama_recipes.modeling.modeling_fast_mistral import MistralForCausalLM
from llama_recipes.modeling.modeling_fast_mistral import MistralDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.datasets import get_streaming_data, get_streaming_sft_data
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies
)


def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    
    if train_config.use_mpi:
        global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
        local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", 0))
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", 1))
        local_world_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])

        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

        env_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"]
        for var in env_vars:
            if var in os.environ:
                print(f"{var} is defined and its value is: {os.environ[var]}")
            else:
                print(f"{var} is not defined.")

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        assert train_config.gradient_accumulation_steps == 1, "Currently doesn't support gradient accumulation with DDP Training"

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    
    if rank == 0:
        print(train_config)
    
    model_class = {
        "llama": LlamaForCausalLM,
        "mistral": MistralForCausalLM,
    }
    model_config = AutoConfig.from_pretrained(train_config.model_name)
    model_type = model_config.model_type
    assert model_type in model_class, f"Model type {model_type} not supported. We currently support llama and mistral."
    
    lastest_checkpoint = train_config.model_name
    if train_config.resume_from_checkpoint:
        lastest_checkpoint = None
        if os.path.isdir(train_config.dist_checkpoint_root_folder):
            checkpoints = [f.path for f in os.scandir(train_config.dist_checkpoint_root_folder) if f.is_dir()]
        
        saved_steps = [int(ckpt_path[ckpt_path.rfind("-")+1: ]) for ckpt_path in checkpoints]
        checkpoint_steps = zip(checkpoints, saved_steps)
        sorted_checkpoints = sorted(checkpoint_steps, key=lambda x: x[1], reverse=True)
        
        lastest_checkpoint = sorted_checkpoints[0][0] # Get the lasest saved version
        assert lastest_checkpoint, f"There is no checkpoint inside the root {train_config.dist_checkpoint_root_folder} folder"
        if rank == 0:
            print(f"Loading checkpoint from {lastest_checkpoint}")
        # model_path = train_config.resume_from_checkpoint
        
    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = model_class[model_type].from_pretrained(
                lastest_checkpoint,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True,
            )
        else:
            model_config.use_cache = use_cache
            with torch.device("meta"):
                model = model_class[model_type](llama_config, use_flash_attention_2=True, torch_dtype=torch.bfloat16,)

    else:
        model = model_class[model_type].from_pretrained(
            lastest_checkpoint,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
        )

    # Load the tokenizer and add special tokens
    # tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, use_fast=True)
    tokenizer.pad_token_id = tokenizer.unk_token_id

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, model_type)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer if model_type == "llama" else MistralDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")
    
    if train_config.use_streaming_data:
        train_dataloader, eval_dataloader = get_streaming_data(train_config)
        if not eval_dataloader:
            train_config.run_validation = False
        
        if train_config.resume_from_checkpoint:
            loader_state_dict = json.load(open(lastest_checkpoint + '/' + f'train_loader_state_dict.json'))
            train_dataloader.load_state_dict(loader_state_dict)
    else:
        dataset_config = generate_dataset_config(train_config, kwargs)

        # Load and preprocess the dataset for training and validation
        dataset_train = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="train",
        )

        if not train_config.enable_fsdp or rank == 0:
            print(f"--> Training Set Length = {len(dataset_train)}")

        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
        if not train_config.enable_fsdp or rank == 0:
                print(f"--> Validation Set Length = {len(dataset_val)}")

        if train_config.batching_strategy == "packing":
            dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

        train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

        # Create DataLoaders for the training and validation dataset
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **train_dl_kwargs,
        )

        eval_dataloader = None
        if train_config.run_validation:
            if train_config.batching_strategy == "packing":
                dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

            val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

            eval_dataloader = torch.utils.data.DataLoader(
                dataset_val,
                num_workers=train_config.num_workers_dataloader,
                pin_memory=True,
                **val_dl_kwargs,
            )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            betas=(0.9, 0.95),
            eps=1e-05,
            weight_decay=0.05,
        )
        
    if train_config.resume_from_checkpoint:
        optimizer_checkpoint_path = lastest_checkpoint + "/" + "optimizer.pt"
        full_osd = None
        if rank == 0:
            print(f"Loading optimizer from {optimizer_checkpoint_path}")
            full_osd = torch.load(optimizer_checkpoint_path)
        # called from all ranks, though only rank0 has a valid param for full_osd
        sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
        optimizer.load_state_dict(sharded_osd)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*train_config.num_epochs, eta_min=1e-5)
    if train_config.resume_from_checkpoint:        
        lr_scheduler_path = lastest_checkpoint + "/" + "lr_scheduler.pt"
        scheduler_state_dict = torch.load(lr_scheduler_path)
        scheduler.load_state_dict(scheduler_state_dict)
    

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    fire.Fire(main)
