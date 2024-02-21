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
import copy
import torch
import random
import argparse
import bittensor as bt
from utils import load_model, save_delta
from data import SubsetFalconLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Main function.
def main( config ):
    
    # Load the model
    model, tokenizer = load_model()
    original_model = copy.deepcopy( model ).cpu()  # Keep a copy of the original model state for delta calculation
    model.to( config.device )

    # Training loop forever.
    epoch = 0
    while True:
        try:
            page = random.randint(0, SubsetFalconLoader.max_pages)
            batches = list(
                SubsetFalconLoader(
                    tokenizer=tokenizer,
                    batch_size=1,
                    sequence_length=512,
                    rows=[page]
                )
            )
        except Exception as e:
            bt.logging.warning(f"Failed to load batches: {e}")
            continue
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
        total_loss = 0.0
        for batch in batches:
            optimizer.zero_grad()
            inputs = batch.to(config.device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        average_loss = total_loss / len(batches)
        bt.logging.success(f"Loss: {average_loss}")
            
        # Save a delta every 2 epochs
        if epoch % 2 == 0: 
            save_delta( model, original_model, config.uid )
            bt.logging.success(f"Saved delta.")

        epoch += 1
            
# Entry point.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and save deltas.")
    parser.add_argument("--uid", required=False, type=int, help="Unique identifier for the delta.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
    bt.logging.add_args( parser )
    config = bt.config( parser )
    main( config )