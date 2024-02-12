import random
import argparse
import bittensor as bt
from tqdm import tqdm

from hparams import batch_size, sequence_length, topk_percent, pages_per_proof
from utils import create_proof, get_model_and_tokenizer, create_seal
from data import SubsetFalconLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_config():
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

def main():
    """
    Main function to run the mining loop with proof creation and seal generation.
    """
    try:
        # Setup configuration
        config = setup_config()
        bt.logging( config = config )
        bt.logging.info("Configuration setup complete.")
        
        # Build Bittensor objects
        wallet = bt.wallet(config=config)
        dendrite = bt.dendrite(wallet=wallet)
        subtensor = bt.subtensor(config=config)
        bt.logging.info("Bittensor objects created.")
        
        # Get model and tokenizer
        model, tokenizer = get_model_and_tokenizer()
        bt.logging.info("Model and tokenizer loaded.")
        
        # Run loop
        while True:
            # Generate random pages
            pages = [random.randint(0, SubsetFalconLoader.max_pages) for _ in range(pages_per_proof)]
            
            # Create proof for the selected pages
            bt.logging.debug("Creating proof for selected pages.")
            proof = create_proof(
                model,
                tokenizer,
                pages=pages,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=config.device,
                topk_percent=topk_percent
            )
            
            # Create seal from the proof
            bt.logging.debug("Creating seal from proof.")
            seal = create_seal(proof, pages, batch_size, sequence_length, topk_percent)
            bt.logging.info(f"Seal created: {seal}")
            
    except Exception as e:
        bt.logging.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
