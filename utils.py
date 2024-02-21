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
import torch
import typing
import hashlib
import bittensor as bt
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def create_model_hash( model: torch.nn.Module ):
    """
    Generates a SHA-256 hash of the model's state dictionary by iterating through the values of each item.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: The SHA-256 hash of the model's state dictionary.
    """
    # Extract the state dictionary from the model which contains all the parameters
    model_state_dict = model.state_dict()
    # Concatenate all the model state values into a single string
    concatenated_model_states = ''.join([ str(value.cpu().numpy().tobytes()) for value in model_state_dict.values()])
    # Encode the concatenated string into bytes
    concatenated_model_states_bytes = concatenated_model_states.encode()
    # Generate a SHA-256 hash from the concatenated bytes
    return hashlib.sha256(concatenated_model_states_bytes).hexdigest()

def save_model(model: torch.nn.Module):
    """
    Saves the current state of the model to disk and logs the time taken to do so.

    This function performs two main tasks:
    1. It saves the state dictionary of the model to a file named 'model.pt' in the 'storage' directory.
    2. It generates a hash for the current state of the model using the `create_model_hash` function,
    and saves this hash to a file named 'model_hash.txt' in the same directory.

    Args:
        model (torch.nn.Module): The model whose state is to be saved.
    """
    # Record the start time to calculate the duration of the save operation.
    start_time = time.time()
    
    # Save the model's state dictionary to 'storage/model.pt'.
    torch.save(model.state_dict(), 'storage/model.pt')
    
    # Generate a hash for the current model state.
    model_hash = create_model_hash(model)
    
    # Save the generated hash to 'storage/model_hash.txt'.
    with open('storage/model_hash.txt', 'w') as f:
        f.write(model_hash)
    
    # Log the duration of the save operation.
    bt.logging.trace(f'Updated model in {time.time() - start_time}s')
    

def load_model() -> torch.nn.Module:
    """
    Loads the GPT2 model and tokenizer from a pre-specified model name.
    Returns the model and tokenizer if successful, logs and returns None otherwise.
    """
    model_name = 'gpt2'
    try:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        if os.path.exists('storage/model.pt'):
            model.load_state_dict(torch.load('storage/model.pt'))
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        bt.logging.debug(f"Model {model_name} loaded with state from storage/model.pt")
        return model, tokenizer
    except Exception as e:
        bt.logging.warning(f'Error while loading model: {e}')
        return None, None

def load_delta(uid: int) -> torch.nn.Module:
    """
    Attempts to load a delta tensor for a given user ID (uid).
    Returns the delta tensor if successful, logs and returns None otherwise.
    """
    try:
        delta_path = f'storage/{uid}/delta.pt'
        delta = torch.load(delta_path, map_location=torch.device('cpu'))
        return delta
    except Exception as e:
        bt.logging.trace(f'Failed to load delta for uid: {uid}, Error: {e}')
        return None
    
def save_delta( model: torch.nn.Module, original_model: torch.nn.Module, uid: int ):
    """
    Saves the delta between the current model state and the original model state.
    
    Args:
        model: The current state of the model.
        original_model: The original state of the model before training.
        uid: Unique identifier for the delta.
    """
    delta = {name: model.state_dict()[name].cpu() - original_model.state_dict()[name].cpu() for name in model.state_dict()}
    delta_path = f"storage/{uid}/delta.pt"
    os.makedirs(os.path.dirname(delta_path), exist_ok=True)
    torch.save(delta, delta_path)
    
def add_delta(model: torch.nn.Module, delta: torch.Tensor):
    """
    Applies a delta to the model parameters by subtracting it.
    Assumes the delta is already scaled by the learning rate.
    """
    for name, param in model.named_parameters():
        if name in delta:
            param.data += delta[name].to(model.device)

def remove_delta(model: torch.nn.Module, delta: torch.Tensor):
    """
    Reverts the changes made by add_delta by adding the delta back to the model parameters.
    """
    for name, param in model.named_parameters():
        if name in delta:
            param.data -= delta[name].to(model.device)

def compute_loss(model: torch.nn.Module, batches, device='cpu') -> float:
    """
    Evaluates the model on a set of batches and returns the average loss.
    """
    with torch.no_grad():  # No need to calculate gradients
        total_loss = 0.0
        num_batches = len(batches)
        for batch in batches:
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            total_loss += outputs.loss.item()
        average_loss = total_loss / num_batches
        return average_loss