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
bt.logging.info("Starting the owner process.")  # Logging the start of the process

# Load the GPT-2 model and tokenizer, and initialize the optimizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)  # Loading the model and moving it to the specified device
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Loading the tokenizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Initializing the optimizer with learning rate 0.001
bt.logging.info(f"Loaded {model_name} model, tokenizer, and initialized optimizer on {config.device}.")  # Logging the successful loading and initialization

# Function to update the saved model state
def update():
    torch.save(model.state_dict(), 'storage/model.pt')  # Saving the model state dictionary
    model_hash = create_model_hash(model)  # Generating a hash for the current model state
    with open('storage/model_hash.txt', 'w') as f:  # Writing the model hash to a file
        f.write(model_hash)
    bt.logging.info("Updated model state and saved to storage.")  # Logging the update
    
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
        for batch in tqdm(batches, desc="Computing Loss"):
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
                
def accumulate_gradient(model, gradient):
    # Apply the gradients to the model
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in gradient: continue
        if param.grad is None: param.grad = torch.zeros_like(param.data, device=device)
        indices, values = gradient[name]
        accumulated_grad = torch.zeros_like(param.grad.view(-1), device=device)
        accumulated_grad.scatter_add_(0, indices.to(torch.long), values)
        param.grad = accumulated_grad.view_as(param.grad)

# Main loop to update the model state and accumulate gradients from miners
global_step = 0  # Initializing the global step counter
n_accs = 2  # Number of accumulations before updating the model
steps_per_loss_calc = 10  # Number of steps before calculating the loss
while True:
    update()  # Updating the model state
    current_model_hash = create_model_hash(model)  # Getting the current model hash
    # Loop to collect gradients until n_accs gradients are collected
    pbar = tqdm(total=n_accs, desc="Accumulating Gradients")  # Initializing tqdm progress bar
    model.zero_grad()  # Resetting gradients of the model
    processed_pages = set()  # Set to store processed pages to avoid duplicates
    for miner, model_hash, page, gradient in yield_gradients():
        if page in processed_pages: continue # Skipping already processed pages
        accumulate_gradient(model, gradient)  # Accumulating the gradients
        pbar.update(1)  # Updating the progress bar for each gradient accumulated
        processed_pages.add(page)
        if len(processed_pages) == n_accs:
            break
    pbar.close()  # Closing the progress bar after loop completion
    optimizer.step()  # Updating the model parameters using the optimizer
    global_step += 1  # Incrementing the global step counter
    if (global_step + 1) % steps_per_loss_calc == 0:
        bt.logging.info(f"Loss: {compute_loss(model, tokenizer, device)}")

