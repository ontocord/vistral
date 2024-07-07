from typing import Dict, Any, List
import torch
import numpy as np
from streaming import StreamingDataset, StreamingDataLoader

def _read_binary_tokenized_sample(sample: Dict[str, Any], max_seq_len: int) -> torch.Tensor:
    return torch.from_numpy(
        np.frombuffer(sample['tokens'], dtype=np.int64)[:max_seq_len].copy())

def intermediate_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int) -> Dict[str, Any]:
    return {'input_ids': torch.stack([_read_binary_tokenized_sample(sample, max_seq_len) for sample in batch])}

def combined_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int = 4096) -> Dict[str, Any]:
    intermediate_result = intermediate_collate_fn(batch, max_seq_len)

    attention_mask = (intermediate_result['input_ids'] != 0).long()

    labels = intermediate_result['input_ids'].clone()

    result = {
        'input_ids': intermediate_result['input_ids'],
        'attention_mask': attention_mask,
        'labels': labels
    }
    return result

def get_streaming_data(train_config):
    train_dataset = StreamingDataset(
        local=train_config.streaming_dataset_path,
        shuffle=True,
        shuffle_seed=42
    )
    
    train_dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=lambda b: combined_collate_fn(b, max_seq_len=train_config.context_length),
    )
    
    eval_dataloader = None #not supporting direct evalation during the training for now
    
    return train_dataloader, eval_dataloader

def get_streaming_sft_data(train_config):
    train_dataset = StreamingDataset(
        local='',
        shuffle=True,
        shuffle_seed=42
    )
    
    train_dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        collate_fn=lambda b: combined_collate_fn(b, max_seq_len=train_config.context_length),
    )
    
    eval_dataloader = None
    if train_config.streaming_eval_data_path:
        eval_dataset = StreamingDataset(local=train_config.streaming_eval_data_path, split=None, shuffle=False)
        eval_dataloader = StreamingDataLoader(
            eval_dataset,
            batch_size=train_config.batch_size_training,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            collate_fn=lambda b: combined_collate_fn(b, max_seq_len=train_config.context_length),
        )
    
    return train_dataloader, eval_dataloader