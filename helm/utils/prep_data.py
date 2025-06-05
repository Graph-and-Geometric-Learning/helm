import os
import json
import time
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from aiohttp.client_exceptions import ClientResponseError  # If your streaming might throw these errors
from transformers import AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--shard_dir",
        type=str,
        required=True,
        help="Directory where chunk shards will be written."
    )
    parser.add_argument(
        "--final_save_path",
        type=str,
        required=True,
        help="Directory for final dataset."
    )
    parser.add_argument("--chunk_size", type=int, default=300_000)
    parser.add_argument("--max_documents", type=int, default=int(1e12)) 
    parser.add_argument("--block_size", type=int, default=2048)
    return parser.parse_args()

def get_next_example(dataset_iter, max_retries=10):

    for retry in range(max_retries):
        try:
            return next(dataset_iter)
        except ClientResponseError as e:
            if e.status in [503, 429]:
                wait_time = 2 ** retry
                print(f"Error {e.status} encountered. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
        except StopIteration:
            return None
    return None

def main():
    args = parse_args()

    wikipedia_config = "20231101.en"
    max_documents = args.max_documents
    chunk_size = args.chunk_size
    shard_dir = args.shard_dir
    final_save_path = args.final_save_path
    access_token = "..." # fill in the access token
    model_name = "meta-llama/Llama-3.1-8B"
    block_size = args.block_size


    os.makedirs(shard_dir, exist_ok=True)


    print(f"Loading Wikipedia dataset for config '{wikipedia_config}' in streaming mode...")
    dataset_stream = load_dataset("wikimedia/wikipedia", wikipedia_config, split="train", streaming=True)
    dataset_iter = iter(dataset_stream)

    collected_docs = 0
    shard_index = 0

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token  
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=True
        )

    def group_texts(examples):
        chunked_input_ids = []
        chunked_attention_masks = []

        for input_ids, attention_mask in zip(examples["input_ids"], examples["attention_mask"]):
            seq_len = len(input_ids)
            if seq_len > block_size:
                total_len = (seq_len // block_size) * block_size
                input_ids = input_ids[:total_len]
                attention_mask = attention_mask[:total_len]
                for i in range(0, total_len, block_size):
                    chunked_input_ids.append(input_ids[i : i + block_size])
                    chunked_attention_masks.append(attention_mask[i : i + block_size])
            else:
                chunked_input_ids.append(input_ids)
                chunked_attention_masks.append(attention_mask)

        return {
            "inputs": chunked_input_ids,
            "masks": chunked_attention_masks
        }

    def shift_labels(example):
        input_ids = torch.tensor(example["input_ids"])
        labels = torch.cat([input_ids[1:], torch.tensor([tokenizer.pad_token_id])])
        labels_list = labels.tolist()
        return {"labels": labels_list}


    data_buffer = []
    while collected_docs < max_documents:
        try:
            example = get_next_example(dataset_iter)
        except StopIteration:
            print("No more examples in the stream.")
            break
        
        if example is None:
            print("Reached dataset end or an unrecoverable streaming error.")
            break
        
        text = example.get("text", "")
        data_buffer.append({"text": text})
        collected_docs += 1

        if len(data_buffer) == chunk_size:
            print(f"Processing chunk: Shard {shard_index}, containing {len(data_buffer)} docs.")
            tmp_dataset = Dataset.from_list(data_buffer)
            tmp_dataset = tmp_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            tmp_dataset = tmp_dataset.map(group_texts, batched=True, remove_columns=tmp_dataset.column_names)
            tmp_dataset = tmp_dataset.rename_column("inputs", "input_ids")
            tmp_dataset = tmp_dataset.rename_column("masks", "attention_mask")
            shard_path = os.path.join(shard_dir, f"wiki_shard_{shard_index}")
            tmp_dataset.save_to_disk(shard_path)
            print(f"Saved shard to {shard_path}.")

            del tmp_dataset
            data_buffer.clear()
            shard_index += 1
        if collected_docs >= max_documents:
            break
    if len(data_buffer) > 0:
        tmp_dataset = Dataset.from_list(data_buffer)
        tmp_dataset = tmp_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tmp_dataset = tmp_dataset.map(group_texts, batched=True, remove_columns=tmp_dataset.column_names)
        tmp_dataset = tmp_dataset.rename_column("inputs", "input_ids")
        tmp_dataset = tmp_dataset.rename_column("masks", "attention_mask")
        
        shard_path = os.path.join(shard_dir, f"wiki_shard_{shard_index}")
        tmp_dataset.save_to_disk(shard_path)
        print(f"Saved final shard to {shard_path}.")

        del tmp_dataset
        data_buffer.clear()
        shard_index += 1

    print(f"Finished collecting {collected_docs} documents.")
    shard_paths = [os.path.join(shard_dir, d) for d in os.listdir(shard_dir) if d.startswith("wiki_shard_")]
    all_shards = []
    for shard_path in sorted(shard_paths):
        ds_shard = Dataset.load_from_disk(shard_path)
        all_shards.append(ds_shard)
    final_dataset = concatenate_datasets(all_shards)
    os.makedirs(final_save_path, exist_ok=True)
    final_dataset.save_to_disk(final_save_path)
    print(f"Final dataset saved to {final_save_path} with {len(final_dataset)} samples.")
    print("Done!")
