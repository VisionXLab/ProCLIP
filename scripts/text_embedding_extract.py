import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import logging
import csv
import time
from tqdm import tqdm
import tarfile
from train.logger import setup_logging

import warnings
import io
from tqdm import tqdm
data_file = {
    'cc3m': 'your_path/cc3m.csv',
    'cc12m': 'your_path/cc12m.csv',
}
from transformers import AutoModel, AutoConfig, AutoTokenizer
from llm2vec import LLM2Vec #here maybe problem, just change package

class my_text_dataset(Dataset):
    def __init__(self,data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        return self.data_list[index][1],self.data_list[index][4],self.data_list[index][5],self.data_list[index][8]  # Assuming the 1th, 4th,  5th and 8th columns are raw caption, s4v_short, s4v_long, and key
    def __len__(self):
        return len(self.data_list)


def load_data(data_name):
    data = []
    data_fl = data_file[data_name]
    with open(data_fl, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in tqdm(reader):
            data.append(row)
    return data

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='cc3m',choices=['cc3m','cc12m']) #for yfcc, you maybe need transfer zip to tar first, we recommend cc3m&cc12m first for your research.
    parser.add_argument('--text_model_dir', default='your_path/llama_cc', help='Model name')#your llama-cc model dir
    parser.add_argument('--batch_size', type=int, default=64, help='Save batch size')
    parser.add_argument('--data_id',type= int,default=1)
    return parser.parse_args()
@torch.no_grad()
def encode_text(args, data_loader):

    config = AutoConfig.from_pretrained(args.text_model_dir, trust_remote_code=True)
    llm_model = AutoModel.from_pretrained(
        args.text_model_dir,
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_dir)
    llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Workaround for LLM2VEC

    model = LLM2Vec(
        llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512
    )
    model.eval()
    root = f'your_path/{args.data_name}_embedding'
    os.makedirs(root, exist_ok=True)
    tar_path_raw_caption = f'your_path/{args.data_name}_embedding/raw_caption_{args.data_id}m.tar'
    tar_path_s4v_short = f'your_path/{args.data_name}_embedding/s4v_short_{args.data_id}m.tar'
    tar_path_s4v_long = f'your_path/{args.data_name}_embedding/s4v_long_{args.data_id}m.tar'

    tar_raw_caption = tarfile.open(tar_path_raw_caption, mode='w')
    tar_short = tarfile.open(tar_path_s4v_short, mode='w')
    tar_long = tarfile.open(tar_path_s4v_long, mode='w')
    for _, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_raw_caption, batch_s4v_short, batch_s4v_long, batch_key = batch
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            raw_caption_embeddings = model.encode(batch_raw_caption,convert_to_tensor=True)
            s4v_short_embeddings = model.encode(batch_s4v_short,convert_to_tensor=True)
            s4v_long_embeddings = model.encode(batch_s4v_long,convert_to_tensor=True)

        for i in range(len(batch_key)):
            key = batch_key[i]
            raw_caption_embedding = raw_caption_embeddings[i].squeeze().clone()
            s4v_short_embedding = s4v_short_embeddings[i].squeeze().clone()
            s4v_long_embedding = s4v_long_embeddings[i].squeeze().clone()

            bio_short = io.BytesIO()
            torch.save(raw_caption_embedding.cpu(), bio_raw_caption)  
            bio_raw_caption.seek(0)  

            bio_short = io.BytesIO()
            torch.save(s4v_short_embedding.cpu(), bio_short)
            bio_short.seek(0)

            bio_long = io.BytesIO()
            torch.save(s4v_long_embedding.cpu(), bio_long)
            bio_long.seek(0)
            tar_raw_caption.addfile(tarinfo=key+'noise.pt', fileobj=bio_raw_caption)
            tar_short.addfile(tarinfo=key + 'short.pt', fileobj=bio_short)
            tar_long.addfile(tarinfo=key + 'long.pt', fileobj=bio_long)

    tar_raw_caption.close()
    tar_short.close()
    tar_long.close()
def main():
    args = parse_args()
    data = load_data(args.data_name)
    seg = len(data) // 8
    if args.data_id < 8:
        rows = data[seg * (args.data_id - 1): seg * args.data_id]
    else:
        rows = data[seg * (args.data_id - 1):]

    dataset_split = my_text_dataset(rows)
    dataloader_split = DataLoader(dataset_split, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    logging.info(f"Number of data: {len(dataset_split)}")
    # assert begin and end 
    logging.info(f'Encoding text data {args.data_name} with model {args.text_model_name} of batch size {args.batch_size}...')
    encode_text(args, dataloader_split)

if __name__ == "__main__":
    llm_model_name = "your_path/llama_cc"
    config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
    llm_model = AutoModel.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Workaround for LLM2VEC

    model = LLM2Vec(
        llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512
    )
    model.eval()
    main()