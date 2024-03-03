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
import random
import argparse
import traceback
import bittensor as bt
import gradient as grad
from transformers import GPT2LMHeadModel

def main(config):
    # Threshold for improvement to save the model
    improvement_threshold = 0.999  
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    grad.utils.push_master( model )
    model.to( config.device )
    while True:
        try:
            # Load a random set of batches
            batches = grad.data.get_random_batches( n = config.pages_per_epoch, batch_size = config.bs, sequence_length = config.sl )
            
            # Compute the base loss for comparison
            base_loss = grad.utils.compute_losses(model, batches, device=config.device)
            bt.logging.success(f"Base score computed for comparison: {base_loss}")
            
            # Load the deltas and compute the loss dif
            for uid, info in grad.utils.list_models().items():
                try:
                    # Load the delta
                    delta = grad.utils.pull_model( uid )
                    if delta is None: continue
                    
                    # Compute the loss after applying the delta
                    grad.utils.add_delta( model, delta )
                    loss = grad.utils.compute_losses(model, batches, device=config.device)
                    bt.logging.info(f"Loss {uid}: {loss}, {loss - base_loss}")
                    
                    # If the loss has improved significantly, save the model
                    if loss < base_loss * improvement_threshold:
                        grad.utils.push_master( model )
                        bt.logging.success(f"Model updated with delta from {uid} given new loss: {loss} < base loss: {base_loss * improvement_threshold}")
                        base_loss = loss
                    else:
                        # Remove the delta which increased the loss.
                        grad.utils.remove_delta( model, delta )

                except Exception as e:
                    continue
                
        except Exception as e:
            bt.logging.error(f"An error occurred during training: \n{e}\n{traceback.format_exc()}")
            continue
        
if __name__ == "__main__":
    # Parse command line arguments for configuration
    parser = argparse.ArgumentParser(description="Train a model and save deltas.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
    parser.add_argument("--bs", default=1, type=int, help="Batch size.")
    parser.add_argument("--sl", default=512, type=int, help="Sequence length")
    parser.add_argument("--pages_per_epoch", default=3, type=int, help="Training pages per epoch.")
    bt.logging.add_args(parser)
    
    # Load the configuration
    config = bt.config(parser)
    bt.logging( config = config )
    
    # Start the main training loop
    main(config)
