# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import glob
import torch
import random
import argparse
import bittensor as bt
from tqdm import tqdm
from utils import topk_gradient, create_model_hash
from data import SubsetFalconLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

parser = argparse.ArgumentParser(description="Validator config")
parser.add_argument("--uid", type=int, default=0, help="Miner UID")
parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu, cuda)")
config = bt.config(parser)
device = torch.device(config.device)
save_path = f'storage/grads/miner_{config.uid}'
os.makedirs(save_path, exist_ok=True)
bt.logging.info(f"Created/verified save path at {save_path}")

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
bt.logging.debug(f"Loaded tokenizer for model {model_name}")

def get_model():
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.load_state_dict(torch.load('storage/model.pt', map_location=device))
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    bt.logging.debug(f"Model {model_name} loaded with state from storage/model.pt and moved to {config.device}")
    return model, tokenizer

def get_model_hash():
    with open('storage/model_hash.txt', 'r') as f:
        model_hash = f.read()
        bt.logging.debug(f"Loaded model hash: {model_hash}")
        return model_hash
    
def clear_grads( hh ):
    grad_files = glob.glob(f'{save_path}/*.pt')
    for grad_file in grad_files:
        if not grad_file.endswith(f"{hh}.pt"):
            os.remove(grad_file)

hash_buffer = []
while True:
    try:
        model, tokenizer = get_model()
        mhash = get_model_hash()
        hash_buffer.append( mhash )

        # Clear hashes out of buffer
        if len(hash_buffer) > 10:
            bt.logging.info(f'clearning grads with hash: {hash_buffer[0]}')
            clear_grads( hash_buffer[0] )
            hash_buffer = hash_buffer[1:]

        while mhash == get_model_hash():
            page = random.randint(0, SubsetFalconLoader.max_pages)
            batches = list(
                SubsetFalconLoader(
                    tokenizer=tokenizer,
                    batch_size=1, 
                    sequence_length=512,
                    rows=[page]
                )
            )
            model.zero_grad()
            for batch in batches:
                batch = batch.to(device)
                model(batch, labels=batch).loss.backward()
            gradient = topk_gradient(model, topk_percent=0.1)
            torch.save(gradient, f'{save_path}/{page}-{mhash}.pt')
            bt.logging.info(f"+ page: {page} hash: {mhash}.pt")
    except Exception as e:
        bt.logging.warning(f'Error during train step: {e}')
