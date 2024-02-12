import time
import typing
import random
import asyncio
import argparse
import bittensor as bt
from protocol import Verify
from hparams import pages_per_proof, topk_percent, sequence_length, batch_size, verify_rate
from utils import get_model_and_tokenizer, create_proof, compute_proof_hash

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
        self.history = {}

        # Setup axon and attach verification and blacklist functions
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.axon.attach(forward_fn=self.verify, blacklist_fn=self.blacklist)
        bt.logging.info("Validator initialized.")
        
    async def blacklist(self, synapse: Verify) -> typing.Tuple[bool, str]:
        """
        Checks if a miner is registered in the metagraph and should be blacklisted.
        
        Args:
            synapse (Verify): The verification request containing the miner's public key.
        
        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether to blacklist the miner,
                              and a string message with the reason.
        """
        if synapse.public_key not in self.metagraph.hotkeys:
            bt.logging.debug(f"Miner {synapse.public_key} not registered in metagraph, blacklisting.")
            return True, 'miner not registered in metagraph'
        return False, 'success'
    
    async def verify(self, synapse: Verify) -> Verify:
        """
        Verifies the proof of work provided by a miner.
        
        Args:
            synapse (Verify): The verification request containing the miner's work to verify.
        
        Returns:
            Verify: The original verification request, potentially modified based on verification result.
        """
        try:
            # Check if the miner's public key is already in the history, if not, initialize it
            if synapse.public_key not in self.history:
                self.history[synapse.public_key] = {'total': 0, 'verified': 0, 'valid': 0}
            miner_history = self.history[synapse.public_key]
            
            # Randomly decide whether to verify or not based on the verify_rate
            if random.random() > verify_rate:
                bt.logging.trace("Skipping verification due to rate.")
                miner_history['total'] += 1
                return synapse
            
            # Perform verification
            bt.logging.debug(f"Verifying proof for miner {synapse.public_key}.")
            proof = create_proof(
                self.model,
                self.tokenizer,
                pages=synapse.pages,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=self.config.device,
                topk_percent=topk_percent
            )
            hash_ = compute_proof_hash(
                proof,
                synapse.pages,
                batch_size,
                sequence_length,
                topk_percent
            )
            if hash_ == synapse.hash:
                bt.logging.info(f"Proof verified for miner {synapse.public_key}.")
                miner_history['total'] += 1
                miner_history['verified'] += 1
                miner_history['valid'] += 1
            else:
                bt.logging.info(f"Proof failed for miner {synapse.public_key}.")
                miner_history['total'] += 1
                miner_history['verified'] += 1
        except Exception as e:
            bt.logging.error(f"Error during verification: {str(e)}")
        finally:
            return synapse
            
    def run(self):
        """
        Starts the Validator's Axon server and begins the verification loop.
        """
        bt.logging.info("Validator run loop started.")
        try:
            bt.logging.info("Serving validator axon.")
            self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
            self.axon.start()
            bt.logging.info("Validator server started.")
            
            while True:
                time.sleep(1)
                bt.logging.debug(f"Current history: {self.history}")
        except Exception as e:
            bt.logging.error(f"Error in run loop: {str(e)}")
            
if __name__ == "__main__":
    try:
        validator = Validator()
        validator.run()
    except Exception as e:
        bt.logging.error(f"Failed to start Validator: {str(e)}")
