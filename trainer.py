"""
The MIT License (MIT)
Copyright © 2023 Yuma Rao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import random
import argparse
import bittensor as bt
from utils import load_model, save_model, compute_loss, load_delta, add_delta
from data import SubsetFalconLoader

def main(config):
    """
    Main training loop.
    
    Args:
        config: Configuration object containing device and other settings.
    """
    bt.logging.info("Starting the main training loop.")
    # Number of uid's to pull deltas from
    n_uids = 3
    bt.logging.debug(f"Number of UIDs to pull deltas from: {n_uids}")
    # Threshold for improvement to save the model
    improvement_threshold = 0.99
    bt.logging.debug(f"Improvement threshold set at: {improvement_threshold}")
    
    while True:
        try:
            bt.logging.info("Attempting to load model and tokenizer.")
            # Load the model and tokenizer
            model, tokenizer = load_model()
            model.to(config.device)
            if model is None or tokenizer is None:
                bt.logging.warning("Model or tokenizer could not be loaded. Exiting loop.")
                raise ValueError("Model or tokenizer could not be loaded.")
            bt.logging.success("Model and tokenizer successfully loaded.")
            
            # Generate a list of random pages to fetch data from
            pages = [random.randint(0, SubsetFalconLoader.max_pages) for _ in range(10)]
            bt.logging.debug(f"Random pages selected for data fetching: {pages}")
            # Load batches of data from the selected pages
            batches = list(SubsetFalconLoader(tokenizer=tokenizer, batch_size=1, sequence_length=512, rows=pages))
            bt.logging.debug("Data batches successfully loaded.")
            
            # Compute the base loss for comparison
            base_score = compute_loss(model, batches, device=config.device)
            bt.logging.debug(f"Base score computed for comparison: {base_score}")
            
            for uid in range(n_uids):
                bt.logging.debug(f"Loading delta for UID: {uid}")
                # Load the delta for the current iteration
                delta = load_delta(uid)
                if delta is None:
                    bt.logging.debug(f"No delta found for UID: {uid}. Skipping.")
                    continue
                
                bt.logging.debug(f"Applying delta for UID: {uid}")
                # Apply the delta to the model
                add_delta(model, delta)
                
                # Compute the loss after applying the delta
                loss = compute_loss(model, batches, device=config.device)
                bt.logging.info(f"Loss {uid}: {loss}")
                
                # If the loss has improved significantly, save the model
                if loss < base_score * improvement_threshold:
                    bt.logging.debug(f"Significant improvement detected for UID {uid}. Saving model.")
                    save_model(model)
                    bt.logging.success("Model updated")
                    
        except Exception as e:
            bt.logging.error(f"An error occurred during training: \n{e}")
            continue
        
if __name__ == "__main__":
    # Parse command line arguments for configuration
    parser = argparse.ArgumentParser(description="Train a model and save deltas.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
    bt.logging.add_args(parser)
    
    # Load the configuration
    config = bt.config(parser)
    bt.logging( config = config )
    
    # Start the main training loop
    main(config)
