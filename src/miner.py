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
import copy
import torch
import random
import argparse
import bittensor as bt
from tqdm import tqdm

from utils import pull_master, hash_model, download_master_hash, calculate_delta, push_model
from dataset import get_random_batches

# Main function.
def main( config ):
    
    # Build Bittensor objects.
    print (config)
    wallet = bt.wallet(config=config)
    bt.logging.success(f"Using wallet: {wallet}")

    # Connect Subtensor.
    subtensor = bt.subtensor(network = 'test')
    bt.logging.success(f"Using subtensor: {subtensor}")

    # Sync metagraph..
    metagraph = subtensor.metagraph( config.netuid )
    bt.logging.success(f"Using subnet: {metagraph}")

    # Check wallet registration.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Miner is not registered, run btcli s register --netuid {config.netuid} --wallet.name {config.wallet.name} --wallet.hotkey {config.wallet.hotkey}')
    else: 
        my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        bt.logging.success(f'Miner is registered on uid: {my_uid} on netuid: {config.netuid}')
        
    # Commit bucket information to chain.
    commit_bucket = subtensor.get_commitment( netuid=config.netuid, uid=my_uid ) 
    if commit_bucket != config.s3_bucket:
        bt.logging.success(f"Advertising bucket: {config.s3_bucket}")
        subtensor.commit(wallet, netuid = config.netuid, data = config.s3_bucket ) 
    bt.logging.success(f"Using bucket: {config.s3_bucket}")
        
    # Check that AWS credentials have been set.
    if 'AWS_ACCESS_KEY_ID' not in os.environ or 'AWS_SECRET_ACCESS_KEY' not in os.environ:
        raise EnvironmentError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")
    bt.logging.success(f"Loaded AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from .env.")

    # Pull initial state.
    master = pull_master()
    if master == None:
        raise ValueError('No master found, wait for the owner to set a master model.')
    model = copy.deepcopy( master )
    bt.logging.success(f"Loaded master model.")

    # Training loop forever.   
    while True:
        
        # If the master model has changed, pull the latest.
        master_hash = hash_model( master )
        if download_master_hash() != master_hash:
            master = pull_master()
            model = copy.deepcopy( master.cpu() )
            bt.logging.success(f"Loaded new master with hash: {master_hash}")

        # Load dataset.
        batches = get_random_batches( n = config.pages_per_epoch, batch_size = config.bs, sequence_length = config.sl )
        
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
            
        # Save delta to bucket.
        delta = calculate_delta( model, master )
        push_model( config.s3_bucket, my_uid, delta )
        bt.logging.success(f"Pushed delta to bucket: {config.s3_bucket} for uid: {my_uid} on netuid: {config.netuid}")
            
# Entry point.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and save deltas.")
    parser.add_argument("--netuid", default=81, type=int, help="Netuid.")
    parser.add_argument("--s3_bucket", default=None, type=str, help="S3 bucket.")
    parser.add_argument("--bs", default=1, type=int, help="Batch size.")
    parser.add_argument("--sl", default=512, type=int, help="Sequence length")
    parser.add_argument("--batches_per_step", default=1, type=int, help="Number of steps before applying a gradient step.")
    parser.add_argument("--pages_per_epoch", default=3, type=int, help="Training pages per epoch.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
    bt.logging.add_args( parser )
    bt.wallet.add_args(parser)
    config = bt.config( parser )
    if config.s3_bucket == None:
        bt.logging.error(f"You must set up your s3 bucket before running this script and pass the name to --s3_bucket [your_bucket_name]")
        exit()
    main( config )
