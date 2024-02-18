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

import random
import torch
import argparse
import bittensor as bt
from data import SubsetFalconLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

parser = argparse.ArgumentParser(description="Validator config")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
parser.add_argument("--netuid", type=int, default=81, help="Netuid to mine on for Bittensor.")
config = bt.config(parser)

model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
def get_model():
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.load_state_dict(torch.load('storage/model.pt'))
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    bt.logging.info(f"Model {model_name} loaded with state from storage/model.pt")
    return model, tokenizer

def get_model_hash():
    with open('storage/model_hash.txt', 'r') as f:
        model_hash = f.read()
        bt.logging.info(f"Loaded model hash: {model_hash}")
        return model_hash
    
def get_delta( uid ):
    delta_path = f'storage/{uid}/delta.pt'
    delta = torch.load(delta_path, map_location=torch.device('cpu'))
    return delta

def apply_delta(model, delta):
    # Iterate through each parameter in the model
    for name, param in model.named_parameters():
        # Check if the parameter is in the delta
        if name in delta:
            # Update the model's parameter by subtracting the delta
            # Assuming the delta is already scaled by the learning rate
            param.data -= delta[name]
            
def score( model, tokenizer, device='cpu'):
    page = random.randint(0, SubsetFalconLoader.max_pages)
    batches = list(
        SubsetFalconLoader(
            tokenizer=tokenizer,
            batch_size=1, 
            sequence_length=512,
            rows=[page]
        )
    )
    with torch.no_grad():  # No need to calculate gradients
        total_loss = 0.0
        num_batches = len(batches)
        for batch in batches:
            total_loss += model(batch, labels=batch).loss.item()
    return total_loss / num_batches
    
scores = {}
while True:
    model = get_model()
    model_hash = get_model_hash()
    for uid in range(3):
        delta = get_delta( uid )
        apply_delta( model, delta )
        scores[ uid ] = score( model, tokenizer, config.device )
