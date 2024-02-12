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
import random
import argparse
import multiprocessing
import bittensor as bt
from tqdm import tqdm

from hparams import batch_size, sequence_length, topk_percent, pages_per_proof
from utils import create_proof, get_model_and_tokenizer, create_seal
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
            
            global_steps = 0
            last_sync_time = time.time()
            last_step_time = time.time()
            total_tokens = 0
            while True:
                tokens_mined = batch_size * sequence_length  # Calculate tokens mined based on batch size and sequence length
                self.mine()
                current_time = time.time()
                step_duration = current_time - last_step_time
                steps_per_second = 1 / step_duration if step_duration > 0 else 0
                tokens_per_second = tokens_mined / step_duration if step_duration > 0 else 0
                total_tokens += tokens_mined
                if current_time - last_sync_time >= 600:  # 600 seconds = 10 minutes
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
        
        # Create proof for the selected pages
        bt.logging.debug("Creating proof for selected pages.")
        proof = create_proof(
            self.model,
            self.tokenizer,
            pages=pages,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=self.config.device,
            topk_percent=topk_percent
        )
        
        # Create seal from the proof
        bt.logging.debug("Creating seal from proof.")
        seal = create_seal(self.model, proof, pages, batch_size, sequence_length, topk_percent)
        bt.logging.info(f"Seal created: {seal}")
        
        # Send seal to validators
        validator_axons = [self.metagraph.axons[uid] for uid in self.metagraph.uids[self.metagraph.validator_permit]]
        self.dendrite.query(validator_axons, seal)

if __name__ == "__main__":
    miner = Miner()
    miner.run()

