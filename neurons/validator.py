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

def main(config):

    wallet = bt.wallet(config=config)
    bt.logging.success(f"Using wallet: {wallet}")
    
    subtensor = bt.subtensor(network = 'test')
    bt.logging.success(f"Using subtensor: {subtensor}")
    
    metagraph = subtensor.metagraph( config.netuid )
    bt.logging.success(f"Using subnet: {metagraph}")
    
    weights = torch.zeros(metagraph.n.item())   
    bt.logging.success(f"Initial weights: {weights}")
    
    # Check wallet registration.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Validator is not registered, run btcli s register --netuid {config.netuid} --wallet.name {config.wallet.name} --wallet.hotkey {config.wallet.hotkey}')
    else: 
        my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        bt.logging.success(f'Validator is registered on uid: {my_uid} on netuid: {config.netuid}')
        
    while True:
        try:
            # Load the model and tokenizer
            master = grad.utils.pull_master()
            master.to(config.device)
            master.eval()

            # Load a random set of batches
            batches = grad.data.get_random_batches( n = config.pages_per_epoch, batch_size = config.bs, sequence_length = config.sl )

            # Compute the base score for comparison
            master_loss = grad.utils.compute_losses(master, batches, device=config.device)
            delta_losses = torch.zeros(metagraph.n.item())
            
            for uid in metagraph.uids:
                try:       
                    # Get bucket name from subtensor commits.             
                    bucket_name = subtensor.get_commitment( config.netuid, uid )
                    if bucket_name is None: 
                        bt.logging.trace(f'uid: { uid }, has no bucket.')
                        continue
                    
                    # Get the delta from the bucket.
                    delta = grad.utils.download_model( bucket_name, uid )
                    if delta is None: 
                        bt.logging.trace(f'uid: { uid }, has no delta.')
                        continue
                    
                    # Apply the delta to the model
                    grad.utils.add_delta(master, delta)
                    
                    # Compute the delta loss with the delta applied.
                    delta_loss = master_loss - grad.utils.compute_losses(master, batches, device=config.device)
                    
                    # Save the loss.
                    delta_losses[uid] = delta_loss
                    
                    # Remove the delta from our local model.
                    grad.utils.remove_delta(master, delta)
                    
                except Exception as e:
                    bt.logging.trace(f'uid: { uid }, failed with error: {e}')   
                    continue
                
            # Weights are the softmax of the loss deltas
            weights = config.alpha * torch.softmax(delta_losses, dim=0) + (1 - config.alpha) * weights
            bt.logging.success(f"weights: {weights}")
            
            # Set weights every 100 blocks.
            current_block = subtensor.get_current_block()
            if current_block % 100 == 0:
                weights.nan_to_num(0.0)
                subtensor.set_weights(
                    netuid = config.netuid,
                    wallet = wallet,
                    uids = metagraph.uids,
                    weights = weights,
                    wait_for_inclusion = False,
                )
            
            # Resync metagraph.
            if current_block % 100 == 0:
                prev_hotkeys = copy.deepcopy( metagraph.hotkeys )
                metagraph = subtensor.metagraph( config.netuid )
                # Clear old scores from newly registered miners.
                for uid, (ha, hb) in enumerate(list(zip( prev_hotkeys, metagraph.hotkeys ))):
                    if ha != hb: weights[uid] = 0.0
                        
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
    config = bt.config(parser)
    bt.logging(config = config)    
    main(config)
