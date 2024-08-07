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
import copy
import boto3
import torch
import typing
import hashlib
import tempfile
import bittensor as bt
from dotenv import dotenv_values
from types import SimpleNamespace
from typing import Optional, Dict, Tuple, List
from botocore.exceptions import BotoCoreError, ClientError
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Define the name of the S3 bucket where the model and its hash will be stored.
env_config = dotenv_values(".env")
MASTER = 8008135
model_name: str = 'gpt2'
storage_location: str = os.path.expanduser('~/.cache')
MASTER_BUCKET: str = 'turingbucket123'
MASTER_UID: int = '0'
if 'AWS_ACCESS_KEY_ID' not in env_config or 'AWS_SECRET_ACCESS_KEY' not in env_config:
    raise Exception("Please provide AWS credentials in the .env file; touch .env and add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

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
            aws_access_key_id=env_config['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=env_config['AWS_SECRET_ACCESS_KEY']
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
    
def download_hash( bucket: str, uid: int ) -> Optional[str]:
    start_time = time.time()
    try:
        s3client: boto3.client = client()  # Create an S3 client
        object_key: str = f'hash_{uid}.txt'
        with tempfile.TemporaryFile() as temp_file:
            s3client.download_fileobj(bucket, object_key, temp_file)
            temp_file.seek(0)  # Go to the start of the file
            module_hash: str = temp_file.read().decode('utf-8')  # Read and decode the hash
        bt.logging.debug(f"Model hash {module_hash} successfully loaded from S3 in {time.time() - start_time}s")
        return module_hash
    except Exception as e:
        bt.logging.trace(f"Failed to load model hash from S3: {e}")
        return None
    
def download_model( bucket: str, uid: int ) -> torch.nn.Module:
    start_time = time.time()
    model_name: str = 'gpt2'
    try:
        # Initialize S3 client to interact with the bucket
        s3client: boto3.client = client( )
        object_key: str = f'module_{uid}.pt'
        # Use a temporary file to store the downloaded model state
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            s3client.download_fileobj(bucket, object_key, temp_file)
            temp_file_path: str = temp_file.name

            # Load the model state dictionary from the temporary file
            model_state_dict: Dict[str, torch.Tensor] = torch.load(temp_file_path)
            model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)
            model.load_state_dict(model_state_dict)
            bt.logging.debug(f"Model {model_name} successfully loaded from S3 bucket {bucket} in {time.time() - start_time}s")

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
        
def upload_model( bucket: str, uid: int, module: torch.nn.Module ):
    try:
        # Check if we should update the model
        current_hash = hash_model( module )
        if download_hash( bucket, uid ) != current_hash:
            # Record the start time to calculate the duration of the save operation.
            start_time: float = time.time()
            # Serialize the model's state dictionary and save it to S3 as 'model.pt'.
            module_state_dict = module.state_dict()
            with io.BytesIO() as module_buffer:
                torch.save(module_state_dict, module_buffer)
                module_buffer.seek(0)
                client().upload_fileobj(module_buffer, bucket, f'module_{uid}.pt')
                bt.logging.debug("Module state dictionary saved to S3.")
                    
            # Save the generated hash to S3 as 'model_hash.txt'.
            with io.BytesIO(current_hash.encode()) as hash_buffer:
                client().upload_fileobj(hash_buffer, bucket, f'hash_{uid}.txt')
                bt.logging.debug("Module hash saved to S3.")
            
            # Log the duration of the save operation.
            bt.logging.success(f'Updated module in {time.time() - start_time}s')
    except Exception as e:
        bt.logging.error(f"Failed to save module: {e}")
        
def download_master_hash() -> str:
    return download_hash( MASTER_BUCKET, MASTER_UID )

def pull_model( bucket: str, uid: int ) -> torch.nn.Module:
    loaded_hash = load_hash( uid )
    downloaded_hash = download_hash( bucket, uid )
    if loaded_hash != downloaded_hash:
        downloaded_model = download_model( bucket, uid )
        save_model( uid, downloaded_model )
        bt.logging.debug(f'Downloaded {uid} from cache {loaded_hash} -> {downloaded_hash} ')
        return downloaded_model
    else:
        bt.logging.debug(f'Loaded {uid} from cache.')
        return load_model( uid )

def push_model( bucket: str, uid: int, model: torch.nn.Module ):
    save_model( uid, model )
    upload_model( bucket, uid, model )
    
def pull_master() -> torch.nn.Module:
    return pull_model( MASTER_BUCKET, MASTER_UID )

def push_master( module: torch.nn.Module ):
    push_model( MASTER_BUCKET, MASTER_UID, module )

def list_models( bucket: str ) -> Dict[int, SimpleNamespace]:
    try:
        s3client: boto3.client = client()
        response = s3client.list_objects_v2(Bucket=bucket, Prefix='hash_')
        deltas_info: Dict[int, Dict[str, typing.Union[str, int]]] = {}

        for obj in response.get('Contents', []):
            try:
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
            except Exception as e:
                bt.logging.trace(f'failed to load object with error: {e}')
                continue
        bt.logging.debug("Successfully listed all deltas from S3.")
        return deltas_info
    except (BotoCoreError, ClientError) as e:
        bt.logging.error(f"Failed to list deltas from S3: {e}")
        raise Exception(f"Failed to list deltas from S3: {e}")

def add_delta(model: torch.nn.Module, delta: torch.nn.Module) -> torch.nn.Module:
    """
    Applies a delta to the model parameters by adding it and returns a new model with the updated parameters.

    This function iterates over the model's parameters and adds the corresponding delta values,
    effectively creating a new model with updated parameters based on the delta.

    Args:
        model (torch.nn.Module): The model to which the delta will be applied.
        delta (torch.nn.Module): The delta to apply to the model's parameters.

    Returns:
        torch.nn.Module: A new model with the delta applied to its parameters.

    Note:
        The delta should be scaled appropriately before being passed to this function.
    """
    try:
        new_model = copy.deepcopy(model)
        model_state_dict = new_model.state_dict()
        delta_state_dict = delta.state_dict()
        for name, param in delta_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name] += param.data.to(new_model.device)
        new_model.load_state_dict(model_state_dict)
        bt.logging.debug("Delta successfully added to the new model.")
        return new_model
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
        model_state_dict = model.state_dict()
        delta_state_dict = delta.state_dict()
        for name, param in delta_state_dict.items():
            if name in model_state_dict:
                model_state_dict[name] -= param.data.to(model.device)
        model.load_state_dict(model_state_dict)
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
    delta_dict = {}
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    for name, param in state_dict_a.items():
        if name in state_dict_b:
            # Subtract the parameters of model_b from model_a and store it in the delta dictionary
            delta_param = param.cpu() - state_dict_b[name].cpu()
            delta_dict[name] = delta_param    
    import copy
    delta = copy.deepcopy( model_a )
    delta.load_state_dict( delta_dict )
    return delta
    