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
import io
import time
import boto3
import torch
import typing
import hashlib
import tempfile
import bittensor as bt
from types import SimpleNamespace
from typing import Optional, Dict, Tuple, List
from botocore.exceptions import BotoCoreError, ClientError
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the name of the S3 bucket where the model and its hash will be stored.
MASTER = 8008135
model_name: str = 'gpt2'
storage_location: str = './storage'
bucket_name: str = 'turingbucket123'

def hash_model(module: torch.nn.Module) -> str:
    """
    Generates a SHA-256 hash of the model's state dictionary.

    This function iterates through the model's state dictionary, concatenates the byte representation
    of each parameter, and then generates a SHA-256 hash of this concatenated byte string.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: The SHA-256 hash of the model's state dictionary.
    """
    try:
        # Extract the state dictionary from the module which contains all the parameters.
        module_state_dict = module.state_dict()
        
        # Concatenate all the model state values into a single byte string.
        concatenated_model_states_bytes = b''.join(
            [value.cpu().numpy().tobytes() for value in module_state_dict.values()]
        )
        
        # Generate a SHA-256 hash from the concatenated bytes.
        module_hash = hashlib.sha256(concatenated_model_states_bytes).hexdigest()
        return module_hash
    except Exception as e:
        bt.logging.error(f"Failed to create model hash: {e}")
        return ""
    
def client():
    """
    Creates and returns an S3 client configured with the specified region and credentials.
    
    Returns:
        boto3.client: An S3 client object.
    """
    try:
        # Create an S3 client with specific AWS region and credentials.
        s3client: boto3.client = boto3.client(
            's3',
            region_name='us-east-1',
            aws_access_key_id='AKIA3TN4TF2QQ4KC4CBA',
            aws_secret_access_key='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'
        )
        return s3client
    except Exception as e:
        bt.logging.error(f"Failed to create S3 client: {e}")
        return None

def model_file_path(uid: int) -> str: 
    return f'{storage_location}/{uid}/model.txt'

def hash_file_path(uid: int) -> str: 
    return f'{storage_location}/{uid}/hash.txt'

def load_hash(uid: int) -> str:
    try:
        with open( hash_file_path( uid ), 'r') as file:
            return file.read()
    except:
        return None
    
def load_model(uid: int) -> torch.nn.Module:
    if not os.path.exists(model_file_path(uid)):
        bt.logging.error(f"Model file does not exist at {model_file_path(uid)}")
        return None
    try:
        model = torch.load(model_file_path(uid), map_location=torch.device('cpu'))
        bt.logging.debug(f"Model loaded successfully from {model_file_path(uid)}")
        return model
    except Exception as e:
        bt.logging.error(f"Failed to load model: {e}")
        return None
    
def download_hash( uid: int ) -> Optional[str]:
    start_time = time.time()
    try:
        s3client: boto3.client = client()  # Create an S3 client
        object_key: str = f'hash_{uid}.txt'
        with tempfile.TemporaryFile() as temp_file:
            s3client.download_fileobj(bucket_name, object_key, temp_file)
            temp_file.seek(0)  # Go to the start of the file
            module_hash: str = temp_file.read().decode('utf-8')  # Read and decode the hash
        bt.logging.debug(f"Model hash {module_hash} successfully loaded from S3 in {time.time() - start_time}s")
        return module_hash
    except Exception as e:
        bt.logging.trace(f"Failed to load model hash from S3: {e}")
        return None
    
def download_model( uid: int ) -> torch.nn.Module:
    start_time = time.time()
    model_name: str = 'gpt2'
    try:
        # Initialize S3 client to interact with the bucket
        s3client: boto3.client = client()
        object_key: str = f'module_{uid}.pt'
        # Use a temporary file to store the downloaded model state
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3client.download_fileobj(bucket_name, object_key, temp_file)
            temp_file_path: str = temp_file.name

            # Load the model state dictionary from the temporary file
            model_state_dict: Dict[str, torch.Tensor] = torch.load(temp_file_path)
            model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)
            model.load_state_dict(model_state_dict)
            bt.logging.debug(f"Model {model_name} successfully loaded from S3 bucket {bucket_name} in {time.time() - start_time}s")

        return model
    except Exception as e:
        bt.logging.debug(f"Failed to load module {e}")
        return None
    
def save_model( uid: int, module: torch.nn.Module ):
    try:
        # Check if we should update the model
        current_hash = hash_model( module )
        if load_hash( uid ) != current_hash:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(model_file_path(uid)), exist_ok=True)
            # Save the model
            torch.save(module, model_file_path(uid))
            bt.logging.debug(f"Model saved successfully at { model_file_path(uid)}")
            # Generate and save the hash
            module_hash = current_hash
            with open(hash_file_path(uid), 'w') as file:
                file.write(module_hash)
        bt.logging.debug(f"Model hash saved successfully at {hash_file_path(uid)}")
    except Exception as e:
        bt.logging.error(f"Failed to save model and hash: {e}")
        
def upload_model( uid: int, module: torch.nn.Module ):
    try:
        # Check if we should update the model
        current_hash = hash_model( module )
        if download_hash( uid ) != current_hash:
            # Record the start time to calculate the duration of the save operation.
            start_time: float = time.time()
            # Serialize the model's state dictionary and save it to S3 as 'model.pt'.
            module_state_dict = module.state_dict()
            with io.BytesIO() as module_buffer:
                torch.save(module_state_dict, module_buffer)
                module_buffer.seek(0)
                client().upload_fileobj(module_buffer, bucket_name, f'module_{uid}.pt')
                bt.logging.debug("Module state dictionary saved to S3.")
                    
            # Save the generated hash to S3 as 'model_hash.txt'.
            with io.BytesIO(current_hash.encode()) as hash_buffer:
                client().upload_fileobj(hash_buffer, bucket_name, f'hash_{uid}.txt')
                bt.logging.debug("Module hash saved to S3.")
            
            # Log the duration of the save operation.
            bt.logging.success(f'Updated module in {time.time() - start_time}s')
    except Exception as e:
        bt.logging.error(f"Failed to save module: {e}")
        
def download_master_hash() -> str:
    return download_hash( MASTER )

def pull_model( uid: int ) -> torch.nn.Module:
    if load_hash( uid ) != download_hash( uid ):
        save_model( uid, download_model( uid ) )
    return load_model( uid )

def push_model( uid: int, model: torch.nn.Module ):
    save_model( uid, model )
    upload_model( uid, model )
    
def pull_master() -> torch.nn.Module:
    return pull_model( MASTER )

def push_master( module: torch.nn.Module ):
    push_model( MASTER, module )

def get_delta_info() -> Dict[str, Dict[str, typing.Union[str, int]]]:
    try:
        s3client: boto3.client = client()
        response = s3client.list_objects_v2(Bucket=bucket_name, Prefix='hash_')
        deltas_info: Dict[int, Dict[str, typing.Union[str, int]]] = {}

        for obj in response.get('Contents', []):
            uid: int = int(obj['Key'].split('_')[1].split('.')[0])  # Extract uid from the object key format "delta_{uid}.pt"
            if uid == MASTER: continue
            local_hash = load_hash( uid )
            remote_hash = download_hash( uid )
            deltas_info[uid] = SimpleNamespace(
                uid=uid,
                module_name=f'module_{uid}.pt',
                hash_name=obj['Key'],
                remote_timestamp=obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S'),
                seconds=int(obj['LastModified'].timestamp()),
                remote=remote_hash,
                local=local_hash,
                stale=remote_hash != local_hash,
            )
        bt.logging.debug("Successfully listed all deltas from S3.")
        return deltas_info
    except (BotoCoreError, ClientError) as e:
        bt.logging.error(f"Failed to list deltas from S3: {e}")
        raise Exception(f"Failed to list deltas from S3: {e}")

def add_delta(model: torch.nn.Module, delta: torch.nn.Module) -> None:
    """
    Applies a delta to the model parameters by adding it.

    This function iterates over the model's parameters and adds the corresponding delta values,
    effectively updating the model's parameters based on the delta.

    Args:
        model (torch.nn.Module): The model to which the delta will be applied.
        delta (torch.nn.Module): The delta to apply to the model's parameters.

    Note:
        The delta should be scaled appropriately before being passed to this function.
    """
    try:
        delta_state_dict = delta.state_dict()
        for name, param in model.named_parameters():
            if name in delta_state_dict:
                param.data += delta_state_dict[name].data.to(model.device)
        bt.logging.debug("Delta successfully added to the model.")
    except Exception as e:
        bt.logging.error(f"Failed to add delta to the model: {e}")
        raise Exception(f"Failed to add delta to the model: {e}")

def remove_delta(model: torch.nn.Module, delta: torch.nn.Module ) -> None:
    """
    Reverts changes made by add_delta by subtracting the delta from the model parameters.

    This function iterates over the model's parameters and subtracts the corresponding delta values,
    effectively reverting the model's parameters to their previous state before the delta was applied.

    Args:
        model (torch.nn.Module): The model from which the delta will be removed.
        delta (torch.nn.Module): The delta to remove from the model's parameters.

    Note:
        This function assumes the delta was previously added to the model using the add_delta function.
    """
    try:
        delta_state_dict = delta.state_dict()
        for name, param in model.named_parameters():
            if name in delta_state_dict:
                param.data -= delta_state_dict[name].data.to(model.device)
        bt.logging.debug("Delta successfully removed from the model.")
    except Exception as e:
        bt.logging.error(f"Failed to remove delta from the model: {e}")
        raise Exception(f"Failed to remove delta from the model: {e}")
    
def calculate_delta(model_a: torch.nn.Module, model_b: torch.nn.Module) -> torch.nn.Module:
    """
    Calculates the difference between two models' parameters and returns it as a torch.nn.Module.

    This function iterates over both models' parameters, computes the difference for each parameter,
    and stores these differences in a new model which is then returned.

    Args:
        model_a (torch.nn.Module): The first model.
        model_b (torch.nn.Module): The second model, to compare against model_a.

    Returns:
        torch.nn.Module: A new model containing the differences between model_a and model_b parameters.
    """
    delta_model = torch.nn.Module()
    for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if name_a == name_b:
            # Ensure the parameter names match, then subtract and store the difference
            setattr(delta_model, name_a, param_a.data - param_b.data)
    return delta_model
    

def compute_losses(model: torch.nn.Module, batches: List[torch.Tensor], device: str = 'cpu') -> float:
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