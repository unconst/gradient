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
import torch
import typing
import random
import asyncio
import argparse
import multiprocessing
import bittensor as bt
from data import SubsetFalconLoader
from protocol import Gradient
from hparams import pages_per_proof, topk_percent, sequence_length, batch_size
from utils import get_model_and_tokenizer, create_model_hash

class Validator:
    """
    A class representing a validator for the Bittensor network.
    
    The validator verifies transactions based on the Bittensor protocol by generating proofs
    of work and comparing them against the hashes provided in transactions.
    """
    
    @staticmethod
    def config() -> bt.config:
        """
        Static method to parse and return the configuration arguments.
        
        Returns:
            bt.config: A configuration object populated with arguments from the command line.
        """
        # Create argument parser for configuration
        parser = argparse.ArgumentParser()
        parser.add_argument("--netuid", type=int, default=81, help="Netuid to mine on.")
        parser.add_argument("--device", type=str, default="cpu", help="Device to use.")
        parser.add_argument("--verify_rate", type=float, default=0.01, help="Rate of verification.")        
        # Add Bittensor-specific arguments to the parser
        bt.axon.add_args(parser)
        bt.wallet.add_args(parser)
        bt.logging.add_args(parser)
        bt.subtensor.add_args(parser)
        
        # Parse arguments to create a configuration object
        config = bt.config(parser)
        return config
    
    def __init__(self, config: bt.config = None):
        """
        Initializes the Validator instance with the given configuration.
        
        Args:
            config (bt.config, optional): The configuration object. If not provided, it will be generated.
        """
        # Setup logging
        bt.logging.info("Initializing Validator...")
        
        # If no config is provided, use the default configuration
        self.config = config or Validator.config()
        
        # Initialize wallet, dendrite, and subtensor
        bt.logging( config = self.config )
        self.wallet = bt.wallet(config=self.config)
        self.dendrite = bt.dendrite(wallet=self.wallet)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(netuid=self.config.netuid)
        
        # Load model and tokenizer
        self.model, self.tokenizer = get_model_and_tokenizer()
        self.model_hash = create_model_hash(self.model)
        self.history = {}

        # Setup axon and attach verification and blacklist functions
        self.axon = bt.axon( wallet=self.wallet, config=self.config )
        self.axon.attach( forward_fn = self.gradient, blacklist_fn=self.blacklist )
        bt.logging.info("Validator initialized.")
        
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
        
    async def blacklist(self, synapse: Gradient) -> typing.Tuple[bool, str]:
        """
        Checks if a miner is registered in the metagraph and should be blacklisted.
        
        Args:
            synapse (Verify): The verification request containing the miner's public key.
        
        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether to blacklist the miner,
                              and a string message with the reason.
        """
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            bt.logging.debug(f"Miner {synapse.public_key} not registered in metagraph, blacklisting.")
            return True, 'miner not registered in metagraph'
        return False, 'success'
    
    async def gradient(self, synapse: Gradient) -> Gradient:
        """
        Verifies the proof of work provided by a miner.
        
        Args:
            synapse (Verify): The verification request containing the miner's work to verify.
        
        Returns:
            Verify: The original verification request, potentially modified based on verification result.
        """
        bt.logging.debug(f"Received gradient from miner {synapse.dendrite.hotkey}.")

        # Deserialize the miner's gradient
        miner_gradient: typing.Dict[ str, typing.Tuple(torch.LongTensor, torch.FloatTensor) ] = {}
        for ( ( key, val ), (_, idx) ) in list(zip(synapse.gradient_vals.items(), synapse.gradient_idx.items())):
            miner_gradient[key] = (idx.tensor(), val.tensor())    
            
        return synapse
    
        # try:
        #     # Check if the miner's public key is already in the history, if not, initialize it
        #     if synapse.dendrite.hotkey not in self.history:
        #         self.history[synapse.dendrite.hotkey] = {'total': 0, 'verified': 0, 'valid': 0, 'failed': 0, 'valid_pages': []}
        #     miner_history = self.history[synapse.dendrite.hotkey]
            
        #     # Deserialize the miner's gradient
        #     miner_gradient: typing.Dict[ str, typing.Tuple(torch.LongTensor, torch.FloatTensor) ] = {}
        #     for ( ( key, val ), (_, idx) ) in list(zip(synapse.gradient_vals.items(), synapse.gradient_idx.items())):
        #         miner_gradient[key] = (idx.tensor(), val.tensor())        
            
        #     # Randomly decide whether to verify or not based on the verify_rate
        #     miner_history['total'] += 1
        #     if random.random() > self.config.verify_rate:
        #         bt.logging.debug("Skipping verification due to rate.")
        #         synapse.vresult = "skipped verification"
        #         return synapse
            
        #     # Check model hash.
        #     if not synapse.model_hash == self.model_hash:
        #         bt.logging.debug(f"Model hash mismatch for miner {synapse.dendrite.hotkey}, got: {synapse.model_hash}, expected: {self.model_hash}.")
        #         synapse.vresult = "invalid model"
        #         return synapse
            
        #     # Check hparams
        #     if synapse.batch_size != batch_size or synapse.sequence_length != sequence_length or synapse.topk_percent != topk_percent:
        #         bt.logging.debug(f"Hparams mismatch for miner {synapse.dendrite.hotkey}.")
        #         synapse.vresult = "invalid hparams"
        #         return synapse
                        
        #     # Create batches of data to process using the SubsetFalconLoader with the given parameters
        #     batches = list(
        #         SubsetFalconLoader(
        #             tokenizer = self.tokenizer,
        #             batch_size = batch_size, 
        #             sequence_length = sequence_length,
        #             rows = synapse.pages
        #         )
        #     )
            
        #     # Move the model to the specified device (CPU or GPU)
        #     self.model.to(self.device)
            
        #     # Reset gradients in the model to zero
        #     self.model.zero_grad()
            
        #     # Process each batch of data
        #     for batch in batches:
        #         # Move the batch to the specified device
        #         inputs = batch.to(self.device)
        #         # Pass the inputs through the model and calculate the loss
        #         outputs = self.model(inputs, labels=inputs)
        #         # Normalize the loss by the number of batches
        #         outputs.loss /= len(batches)
        #         # Backpropagate the loss to compute gradients
        #         outputs.loss.backward()
        #         # Exit the loop after processing the first batch for demonstration purposes
        #         break
            
        #     # Extract the top-k percent gradients from the model
        #     gradient = topk_gradient( self.model, topk_percent )
            
        #     if gradient_hash == synapse.gradient_hash:
        #         bt.logging.debug(f"Proof verified for miner {synapse.dendrite.hotkey}.")
        #         miner_history['verified'] += 1
        #         miner_history['valid'] += 1
        #         miner_history['valid_pages'].extend( synapse.pages )
        #         synapse.vresult = "succeeded verification"
        #     else:
        #         bt.logging.debug(f"Proof failed for miner {synapse.dendrite.hotkey}, got: {synapse.gradient_hash} expected: {gradient_hash}.")
        #         miner_history['verified'] += 1
        #         synapse.vresult = "failed verification"

        # except Exception as e:
        #     bt.logging.debug(f"Error during verification: {str(e)}")
        #     synapse.vresult = "error during verification"

        # finally:
        #     return synapse
            
    def run(self):
        """
        Starts the Validator's Axon server and begins the verification loop.
        """
        bt.logging.info("Validator run loop started.")
        try:
            # Check if the validator is registered and the subnet exists.
            if not self.subtensor.subnet_exists(self.config.netuid):
                raise ValueError(f"Subnet: {self.config.netuid} does not exist on network: {self.config.subtensor.network}\n")
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                raise ValueError(f"Validator is not registered to subnet {self.config.netuid}. Please register first.\n")
            
            bt.logging.info("Serving validator axon.")
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
            self.axon.start()
            bt.logging.info("Validator server started.")
            
            last_sync_time = time.time()
            while True:
                time.sleep(5)
                current_time = time.time()
                if current_time - last_sync_time >= 60:  # Sync every 60 seconds
                    self.try_sync_metagraph(ttl=60)
                    last_sync_time = current_time
                bt.logging.debug(f"Current history: {self.history}")
        except Exception as e:
            bt.logging.error(f"Error in run loop: {str(e)}")
            
if __name__ == "__main__":
    try:
        validator = Validator()
        validator.run()
    except Exception as e:
        bt.logging.error(f"Failed to start Validator: {str(e)}")
