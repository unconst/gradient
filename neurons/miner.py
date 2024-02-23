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

import copy
import torch
import random
import argparse
import bittensor as bt
import gradient as grad
from tqdm import tqdm

# Main function.
def main( config ):
    
    master = grad.utils.pull_master()
    if master == None:
        raise ValueError('No master found.')
    model = copy.deepcopy( master )
    
    # Training loop forever.   
    while True:
        
        # If the master model has changed, pull the latest.
        if grad.utils.download_master_hash() != grad.utils.hash_model( master ):
            master = grad.utils.pull_master()
            model = copy.deepcopy( master.cpu() )
            bt.logging.success(f"Loaded new master.")

        # Load dataset.
        batches = grad.data.get_random_batches( n = config.pages_per_epoch, batch_size = config.bs, sequence_length = config.sl )
        
        # Train model for epoch.
        model.train()
        model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        total_loss = 0.0
        steps = 0
        for batch in tqdm(batches):
            inputs = batch.to(config.device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            # Gradient accumulation.
            if steps % config.batches_per_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            steps += 1
        average_loss = total_loss / len(batches)
        bt.logging.success(f"Loss: {average_loss}")
            
        # Save delta.
        delta = grad.utils.calculate_delta( model, master )
        grad.utils.push_model( config.uid, delta )
        bt.logging.success(f"Pushed delta.")
            
# Entry point.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and save deltas.")
    parser.add_argument("--uid", required=False, type=int, help="Unique identifier for the delta.")
    parser.add_argument("--bs", default=1, type=int, help="Batch size.")
    parser.add_argument("--sl", default=512, type=int, help="Sequence length")
    parser.add_argument("--batches_per_step", default=1, type=int, help="Number of steps before applying a gradient step.")
    parser.add_argument("--pages_per_epoch", default=3, type=int, help="Training pages per epoch.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
    bt.logging.add_args( parser )
    config = bt.config( parser )
    main( config )