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
import torch
import random
import argparse
import bittensor as bt
from tqdm import tqdm
from gradient.utils import pull_master, compute_losses, add_delta, remove_delta, pull_delta, get_delta_info
from gradient.data import get_random_batches

def main(config):

    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(network = 'test')
    metagraph = subtensor.metagraph( config.netuid )
    weights = torch.zeros(metagraph.n.item())   
        
    while True:
        try:
            # Load the model and tokenizer
            model = pull_master()
            model.to(config.device)
            model.eval()

            # Load a random set of batches
            batches = get_random_batches( n = config.pages_per_epoch, batch_size = config.bs, sequence_length = config.sl )

            # Compute the base score for comparison
            base_score = compute_losses(model, batches, device=config.device)
            for info in get_delta_info():
                try:
                    
                    delta = pull_delta( info.uid )
                    if delta is None: continue
                    
                    add_delta(model, delta)
                    delta_loss = compute_losses(model, batches, device=config.device)
                    score_i = base_score - delta_loss
                    scores[info.uid] = config.alpha * score_i + (1 - config.alpha) * scores[info.uid]
                    remove_delta(model, delta)
                    bt.logging.info(f'uid: {info.uid}, base_loss: {base_score}, delta_loss: {delta_loss} score_i: {score_i}, scores[uid]: {weights[info.uid]}')   
                except Exception as e:
                    bt.logging.info(f'uid: {info.uid}, failed.')   
                    continue

            bt.logging.success(f"Scores updated: {weights}")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e}")
            break
    
if __name__ == "__main__":
    # Parse command line arguments for configuration
    parser = argparse.ArgumentParser(description="Validator config")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
    parser.add_argument("--bs", default=1, type=int, help="Batch size.")
    parser.add_argument("--sl", default=512, type=int, help="Sequence length")
    parser.add_argument("--pages_per_epoch", default=3, type=int, help="Training pages per epoch.")
    parser.add_argument("--alpha", type=float, default=0.01, help="Exponential moving average weight.")
    parser.add_argument("--netuid", default=81, type=int, help="Netuid.")
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    
    # Load the configuration
    config = bt.config(parser)
    bt.logging(config = config)
    
    # Start the main evaluation loop
    main(config)
