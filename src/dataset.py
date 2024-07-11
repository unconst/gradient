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
import math
import torch
import typing
import random
import requests
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import GPT2Tokenizer

def compute_losses(model: torch.nn.Module, batches: typing.List[torch.Tensor], device: str = 'cpu') -> float:
    """
    Computes and returns the average loss of a model evaluated over a given set of batches.

    This function iterates through each batch, feeds it to the model, and accumulates the loss to compute
    the average loss across all batches. This is useful for evaluating the model's performance on a dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        batches (List[torch.Tensor]): A list of batches to evaluate the model on. Each batch is a torch.Tensor.
        device (str, optional): The device (e.g., 'cpu' or 'cuda') on which to perform the computations. Defaults to 'cpu'.

    Returns:
        float: The average loss computed over all the batches.

    Note:
        This function does not compute gradients and is typically used for model evaluation.

    Raises:
        ValueError: If `batches` is empty, raising a ValueError to indicate that no batches were provided for evaluation.
    """
    # Ensure there are batches to compute the loss on
    if not batches:
        bt.logging.error("No batches provided for loss computation.")
        raise ValueError("No batches provided for loss computation.")

    # Initialize total_loss to accumulate losses over batches
    total_loss: float = 0.0

    # Calculate the number of batches for averaging the loss later
    num_batches: int = len(batches)

    # Disable gradient computations for efficiency and to prevent model updates
    with torch.no_grad():
        for batch in batches:
            try:
                # Move the batch to the specified device (e.g., CPU or GPU)
                inputs: torch.Tensor = batch.to(device)
                # Forward pass: Compute the model's output and loss for the given inputs
                outputs = model(inputs, labels=inputs)
                # Accumulate the loss
                total_loss += outputs.loss.item()
            except Exception as e:
                bt.logging.error(f"Error during loss computation for a batch: {e}")
                raise Exception(f"Error during loss computation for a batch: {e}")

    # Compute the average loss across all batches
    try:
        average_loss: float = total_loss / num_batches
    except ZeroDivisionError as e:
        bt.logging.error("Division by zero encountered while computing average loss. This should not happen.")
        raise ZeroDivisionError("Division by zero encountered while computing average loss.")

    # Log the computed average loss
    bt.logging.debug(f"Average loss computed successfully: {average_loss}")

    return average_loss

def get_random_batches( n: int, batch_size: int, sequence_length: int ) -> typing.List[torch.FloatTensor]:
    return list( SubsetFalconLoader( batch_size, sequence_length, [ random.randint(0, SubsetFalconLoader.max_pages) for _ in range( n ) ] ) )

class SubsetFalconLoader(IterableDataset):
    max_pages: int = 968000015

    def __init__(self, batch_size, sequence_length, pages: typing.List[int]):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_rows_per_page = 100
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": "tiiuae/falcon-refinedweb",
            "config": "default",
            "split": "train",
        }
        self.pages = pages
        self.buffer = []
        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries

        for page in self.pages:
            self.fetch_data_for_page(page)            

    def fetch_data_for_page(self, page):
        self.params["offset"] = page
        self.params["limit"] = self.num_rows_per_page
        attempt = 0
        while attempt < self.retry_limit:
            try:
                response = requests.get(self.base_url, params=self.params)
                response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
                for row in response.json()["rows"]:
                    content = row["row"]["content"]
                    self.buffer += self.tokenizer(content, truncation=True)["input_ids"]
                    self.buffer += [self.tokenizer.eos_token_id]
                break  # If the request was successful, break out of the retry loop
            except requests.exceptions.RequestException as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to fetch data, retrying. Attempt {attempt}/{self.retry_limit}"
                )
                if attempt < self.retry_limit:
                    time.sleep(self.retry_delay)  # Wait before the next retry
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to fetch data."
                    )
                    raise

    def __iter__(self):
        while len(self.buffer) >= self.sequence_length * self.batch_size:
            batch = []
            for _ in range(self.batch_size):
                batch.append(torch.tensor(self.buffer[: self.sequence_length]))
                self.buffer = self.buffer[self.sequence_length :]
            yield torch.stack(batch)

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(torch.tensor(self.buffer[: self.sequence_length]))
            self.buffer = self.buffer[self.sequence_length :]
        return torch.stack(batch)