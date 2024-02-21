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
import os
import torch
import random
import argparse
import bittensor as bt
from utils import load_model, compute_loss, load_delta, add_delta, remove_delta
from data import SubsetFalconLoader

def main(config):
    """
    Main evaluation loop for validating model deltas.
    
    Args:
        config: Configuration object containing device and other settings.
    """
    # Number of deltas to evaluate
    n = 3
    # Weight for the exponential moving average
    alpha = 0.01
    # Initialize scores dictionary
    scores = {i: 0 for i in range(n)}
    
    bt.logging.info("Starting the validation loop.")
    
    while True:
        try:
            # Load the model and tokenizer
            model, tokenizer = load_model()
            model.to(config.device)
            model.eval()

            if model is None or tokenizer is None:
                bt.logging.error("Model or tokenizer could not be loaded. Exiting.")
                break

            # Load a random set of batches
            try:
                page = random.randint(0, SubsetFalconLoader.max_pages)
                batches = list(SubsetFalconLoader(tokenizer=tokenizer, batch_size=1, sequence_length=512, rows=[page]))
                bt.logging.debug(f"Loaded batches from page {page}.")
            except Exception as e:
                bt.logging.error(f"Failed to load batches: {e}")
                continue

            # Compute the base score for comparison
            base_score = compute_loss(model, batches, device=config.device)
            bt.logging.trace(f"Base score computed: {base_score}")

            # Evaluate all the deltas
            for uid in range(n):
                delta = load_delta(uid)
                if delta is not None:
                    add_delta(model, delta)
                    score_i = base_score - compute_loss(model, batches, device=config.device)
                    scores[uid] = alpha * score_i + (1 - alpha) * scores[uid]
                    remove_delta(model, delta)
                    bt.logging.debug(f"Updated score for uid {uid}: {scores[uid]}")
                else:
                    bt.logging.debug(f"No delta found for uid: {uid}")

            bt.logging.success(f"Scores updated: {scores}")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e}")
            break
    
if __name__ == "__main__":
    # Parse command line arguments for configuration
    parser = argparse.ArgumentParser(description="Validator config")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
    bt.logging.add_args(parser)
    
    # Load the configuration
    config = bt.config(parser)
    bt.logging(config = config)
    
    # Start the main evaluation loop
    main(config)
