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
import time
import glob
import torch
import random
import argparse
import bittensor as bt
from tqdm import tqdm  # Importing tqdm for progress bar functionality

from data import SubsetFalconLoader
from utils import topk_gradient, create_model_hash, accumulate_gradient
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Setting up the argument parser for configuration
parser = argparse.ArgumentParser(description="Owner config")
parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu, cuda)")
bt.logging.add_args(parser)  # Adding bittensor logging arguments to the parser
config = bt.config(parser)  # Parsing the arguments to get the configuration
device = torch.device(config.device)  # Setting the device
bt.logging( config = config )
bt.logging.info("Starting the owner process.")  # Logging the start of the process

# Load the GPT-2 model and tokenizer, and initialize the optimizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)  # Loading the model and moving it to the specified device
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Loading the tokenizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Initializing the optimizer with learning rate 0.001
bt.logging.info(f"Loaded {model_name} model, tokenizer, and initialized optimizer on {config.device}.")  # Logging the successful loading and initialization
model.to(device)

# Function to update the saved model state
def update():
    start_time = time.time()
    torch.save(model.state_dict(), 'storage/model.pt')  # Saving the model state dictionary
    model_hash = create_model_hash(model)  # Generating a hash for the current model state
    with open('storage/model_hash.txt', 'w') as f:  # Writing the model hash to a file
        f.write(model_hash)
    bt.logging.success(f'updated model in {time.time() - start_time}s')
    
def compute_loss(model, tokenizer, device='cpu'):
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
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            total_loss += outputs.loss.item()
    return total_loss / num_batches

def yield_gradients(model_hash: str = "*", miner_uid: str = "*"):
    grad_files_pattern = f'storage/grads/miner_{miner_uid}/*-{model_hash}.pt'
    while True:
        grad_files = glob.glob(grad_files_pattern)
        for grad_file in grad_files:
            file_name = grad_file.split('/')[-1]
            page_number, model_hash = file_name.split('-')
            model_hash = model_hash.split('.')[0]
            miner_uid = grad_file.split('/')[-2].split('_')[1]
            if os.path.exists(grad_file):
                gradients = torch.load(grad_file, map_location=device)
                yield miner_uid, model_hash, page_number, gradients
                
def accumulate_gradient(model, gradient, device):
    # Apply the gradients to the model
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in gradient: continue
        if param.grad is None: param.grad = torch.zeros_like(param.data, device=device)
        indices, values = gradient[name]
        accumulated_grad = torch.zeros_like(param.grad.view(-1), device=device)
        accumulated_grad.scatter_add_(0, indices.to(torch.long), values.to(device))
        param.grad = accumulated_grad.view_as(param.grad)

def load_file( gfile, device = device):
    file_name = grad_file.split('/')[-1]
    page_number, model_hash = file_name.split('-')
    model_hash = model_hash.split('.')[0]
    miner_uid = grad_file.split('/')[-2].split('_')[1]
    return page_number, miner_uid, model_hash

# Main loop to update the model state and accumulate gradients from miners
global_step = 0  # Initializing the global step counter
processed_pages = set()  # Set to store processed pages to avoid duplicates
accs_per_update = 10
hash_history = []
while True:
    if global_step % accs_per_update == 0:
        update()
        current_hash = create_model_hash( model )
        hash_history.append( current_hash )

    bt.logging.success(f"Step {global_step}: {current_hash}" )
    model.zero_grad()  # Resetting gradients of the model
    applied_gradient = False
    
    start_time = time.time()
    while not applied_gradient:
        try:
            n_total = 0
            n_available = 0
            n_stale = 0
            n_used = 0
            grad_files_pattern = f'storage/grads/miner_*/*-*.pt'
            grad_files = glob.glob(grad_files_pattern)
            for grad_file in grad_files:
                n_total += 1
                page, uid, mhash = load_file( grad_file )
                if page in processed_pages: 
                    n_used += 1
                    bt.logging.trace('page already processed'); 
                    continue
                if mhash not in hash_history[-3:]: 
                    n_stale += 1
                    bt.logging.trace('gradient stale'); 
                    continue
                n_available += 1
                if not applied_gradient:
                    apply_time = time.time()
                    grad = torch.load(grad_file, map_location = device)
                    accumulate_gradient( model, grad, device = device )
                    processed_pages.add(page)
                    applied_gradient = True
                    bt.logging.success(f'accumulated gradient:{page} in {time.time() - apply_time}s')
        except Exception as e:
            bt.logging.warning(f'error during accumulation step: {e}')
        bt.logging.success(f'n_total: {n_total}, n_available: {n_available}, n_stale: {n_stale}, n_used: {n_used}')
        time.sleep(1)
    bt.logging.debug(f'applied gradient in {time.time() - start_time}s')
    optimizer.step()  # Updating the model parameters using the optimizer
    global_step += 1  # Incrementing the global step counter
    if global_step % 10 == 0:
        loss_time = time.time()
        bt.logging.success( f"Loss: {compute_loss(model, tokenizer, device)} in {time.time() - loss_time}s")

