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

import time
import typing
import random
import argparse
import multiprocessing
import bittensor as bt
from tqdm import tqdm

from protocol import Gradient
from hparams import batch_size, sequence_length, topk_percent, pages_per_proof
from utils import get_model_and_tokenizer, topk_gradient
from data import SubsetFalconLoader

class Miner:
    def __init__(self):
        """
        Initializes the miner with necessary configurations and objects.
        """
        self.config = self.setup_config()
        bt.logging(self.config)
        self.wallet = bt.wallet(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)

    def setup_config(self):
        """
        Sets up and parses configuration from command line arguments.

        Returns:
            A configuration object with all necessary parameters for running the miner.
        """
        parser = argparse.ArgumentParser(description="Bittensor Miner Configuration")
        parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
        parser.add_argument("--netuid", type=int, default=81, help="Netuid to mine on for Bittensor.")
        bt.wallet.add_args(parser)
        bt.logging.add_args(parser)
        bt.subtensor.add_args(parser)
        config = bt.config(parser)
        return config
    
    def try_sync_metagraph(self, ttl: int):
        """
        Attempts to synchronize the metagraph within a given time-to-live (TTL) period.
        Args:
            ttl (int): The time-to-live (TTL) in seconds for the synchronization attempt.
        """
        def sync_metagraph(endpoint):
            metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid)
            metagraph.save()
        process = multiprocessing.Process(target=sync_metagraph, args=(self.subtensor.chain_endpoint,))
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return
        # Load new state.
        self.metagraph.load()
        
    def run(self):
        """
        Main method to run the mining loop with proof creation and seal generation.
        """
        try:
            bt.logging.info(self.config)
            bt.logging.info("Configuration setup complete.")
            bt.logging.info("Bittensor objects created.")

            # Check if the miner is registered and the subnet exists.
            if not self.subtensor.subnet_exists(self.config.netuid):
                raise ValueError(f"Subnet: {self.config.netuid} does not exist on network: {self.config.subtensor.network}\n")
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                raise ValueError(f"Miner is not registered to subnet {self.config.netuid}. Please register first.\n")

            # Get model and tokenizer
            self.model, self.tokenizer = get_model_and_tokenizer()
            bt.logging.info("Model and tokenizer loaded.")
            
            # Run the mining loop
            global_steps, total_tokens = 0, 0
            last_sync_time = last_step_time = time.time()
            while True:
                self.mine()
                current_time = time.time()
                tokens_mined = batch_size * sequence_length
                total_tokens += tokens_mined
                
                step_duration = current_time - last_step_time
                steps_per_second = 1 / step_duration if step_duration > 0 else 0
                tokens_per_second = tokens_mined / step_duration if step_duration > 0 else 0
                
                if current_time - last_sync_time >= 600:  # Sync every 10 minutes
                    self.try_sync_metagraph(ttl=60)
                    last_sync_time = current_time
                
                global_steps += 1
                bt.logging.success(f"Global step {global_steps}, Steps/s: {steps_per_second:.2f}, Tokens/s: {tokens_per_second:.2f}, Total tokens: {total_tokens}")
                last_step_time = current_time
        except Exception as e:
            bt.logging.error(f"Error occurred: {str(e)}")

    def mine(self):
        """
        Executes a single iteration of the mining process.
        """
        # Generate random pages
        pages = [random.randint(0, SubsetFalconLoader.max_pages) for _ in range(pages_per_proof)]
                
        # Create batches of data to process using the SubsetFalconLoader with the given parameters
        batches = list(
            SubsetFalconLoader(
                tokenizer = self.tokenizer,
                batch_size = batch_size, 
                sequence_length = sequence_length,
                rows = pages
            )
        )
        
        # Move the model to the specified device (CPU or GPU)
        self.model.to(self.config.device)
        
        # Reset gradients in the model to zero
        self.model.zero_grad()
        
        # Process each batch of data
        for batch in batches:
            # Move the batch to the specified device
            inputs = batch.to(self.config.device)
            # Pass the inputs through the model and calculate the loss
            outputs = self.model(inputs, labels=inputs)
            # Normalize the loss by the number of batches
            outputs.loss /= len(batches)
            # Backpropagate the loss to compute gradients
            outputs.loss.backward()
            # Exit the loop after processing the first batch for demonstration purposes
            break
        
        # Extract the top-k percent gradients from the model
        gradient = topk_gradient( self.model, topk_percent )
        
        # Serialize the gradient to a bt.Tensors
        grad_idx: typing.Dict[ str, bt.Tensor ] = {}
        grad_vals: typing.Dict[ str, bt.Tensor ] = {}
        for key in gradient:
            grad_idx[key] = bt.Tensor.serialize( gradient[key][0] )
            grad_vals[key] = bt.Tensor.serialize( gradient[key][1] )
        synapse = Gradient(
            pages = pages,
            gradient_idx = grad_idx,
            gradient_vals = grad_vals,
        )
        size_in_mb = synapse.get_total_size() / (1024 * 1024)
        bt.logging.info(f"Synapse created with size: {size_in_mb} MB")
        # Send seal to validators
        validator_axons = [self.metagraph.axons[uid] for uid in self.metagraph.uids[self.metagraph.validator_permit]]
        responses = self.dendrite.query(validator_axons, synapse, timeout = 1 )

if __name__ == "__main__":
    miner = Miner()
    miner.run()

