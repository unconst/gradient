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

# Import necessary libraries
import torch
import typing
import pickle
import hashlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data import SubsetFalconLoader
from protocol import Gradient

def get_model_and_tokenizer():
    """
    Initializes and returns a GPT-2 model along with its tokenizer.

    Returns:
        tuple: A tuple containing the GPT2LMHeadModel and GPT2Tokenizer instances.
    """
    # Define the model name as 'gpt2' which is a pre-trained model available in the transformers library
    model_name = "gpt2"
    # Load the GPT-2 model from the pre-trained 'gpt2' model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # Load the tokenizer that corresponds to the 'gpt2' model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Return both the model and tokenizer as a tuple
    return model, tokenizer

def topk_gradient(model: torch.nn.Module, topk_percent: float) -> typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extracts the top-k percent gradients of a model's parameters.

    Args:
        model (torch.nn.Module): The model from which to extract gradients.
        topk_percent (float): The percentage of top gradients to extract.

    Returns:
        Dict[str, Tuple[torch.Tensor, torch.Tensor]]: A dictionary mapping parameter names to tuples of indices and values of top-k gradients.
    """
    # Initialize an empty dictionary to store gradient data
    gradient_data = {}
    # Iterate through each parameter in the model
    for name, parameter in model.named_parameters():
        # Check if the parameter has gradients
        if parameter.grad is not None:
            # Calculate the total number of elements in the gradient
            total_elements = parameter.grad.numel()
            # Calculate the number of elements to consider as top-k based on the given percentage
            topk_elements = int(total_elements * topk_percent)
            
            # Extract the top-k values and their indices from the flattened gradient tensor
            values, indices = torch.topk(parameter.grad.abs().flatten(), topk_elements)
            
            # Convert indices and values to the desired data type
            indices = indices.to(torch.int32)
            values = values.to(torch.float32)
            
            # Store the indices and values in the gradient_data dictionary with the parameter name as the key
            gradient_data[name] = (indices, values)
    # Return the dictionary containing the top-k gradient data
    return gradient_data

def accumulate_proofs(model, proofs):
    """
    Accumulates multiple gradient proofs onto a model's gradients.

    Args:
        model (torch.nn.Module): The model to update.
        proofs (List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]): The proofs to accumulate.

    """
    # Iterate through each parameter in the model
    for name, param in model.named_parameters():
        # Check if the parameter requires gradients and has not been processed yet
        if not param.requires_grad: continue
        if param.grad is None: param.grad = torch.zeros_like(param.data)
        
        # Flatten the gradient tensor for easier manipulation
        grad_flat = param.grad.view(-1)
        
        # Iterate through each proof
        for proof in proofs:
            # Check if the current parameter is present in the proof
            if name not in proof: continue
            # Retrieve the indices and values from the proof
            indices, values = proof[name]
            
            # Add the values from the proof to the corresponding indices in the gradient tensor
            grad_flat.scatter_add_(0, indices.to(torch.long), values)
        
        # Reshape the gradient tensor back to its original shape
        param.grad = grad_flat.view_as(param.grad)


        
# def create_model_hash(model):
#     """
#     Generates a SHA-256 hash of the model's state dictionary by iterating through the values of each item.

#     Args:
#         model (torch.nn.Module): The model to hash.

#     Returns:
#         str: The SHA-256 hash of the model's state dictionary.
#     """
#     # Extract the state dictionary from the model which contains all the parameters
#     model_state_dict = model.state_dict()
#     # Concatenate all the model state values into a single string
#     concatenated_model_states = ''.join([ str(value.cpu().numpy().tobytes()) for value in model_state_dict.values()])
#     # Encode the concatenated string into bytes
#     concatenated_model_states_bytes = concatenated_model_states.encode()
#     # Generate a SHA-256 hash from the concatenated bytes
#     return hashlib.sha256(concatenated_model_states_bytes).hexdigest()

# def create_gradient_hash(gradient: typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]]) -> str:
#     """
#     Computes a SHA-256 hash of the gradient proof.

#     Args:
#         gradient (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): The gradient proof to hash.

#     Returns:
#         str: The SHA-256 hash of the gradient proof.
#     """
#     # Concatenate all the gradient values into a single string
#     concatenated_tensors = ''.join([ str(item[1].cpu().numpy().tobytes()) for item in gradient.values()])
#     # Encode the concatenated string into bytes
#     concatenated_tensors = concatenated_tensors.encode()
#     # Generate a SHA-256 hash from the concatenated bytes
#     return hashlib.sha256(concatenated_tensors).hexdigest()


# def create_gradient( 
#         model: torch.nn.Module,
#         tokenizer: 'tokenizer',
#         pages: typing.List[int], 
#         batch_size: int, 
#         sequence_length: int,
#         device: str = 'cpu',
#         topk_percent: float = 0.01
#     ) -> typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]]:
#     """
#     Generates a gradient based on the top-k percent of the model's parameters.

#     Args:
#         model (torch.nn.Module): The model for gradient computation.
#         tokenizer: The tokenizer for data processing.
#         pages (List[int]): The pages to process.
#         batch_size (int): The batch size for processing.
#         sequence_length (int): The sequence length for processing.
#         device (str): The computation device ('cpu' or 'gpu').
#         topk_percent (float): The percentage of top gradients to retain.

#     Returns:
#         Dict[str, Tuple[torch.Tensor, torch.Tensor]]: The top-k gradients.
#     """
#     # Move the model to the specified device (CPU or GPU)
#     model.to(device)
    
#     # Create batches of data to process using the SubsetFalconLoader with the given parameters
#     batches = list(
#         SubsetFalconLoader(
#             tokenizer=tokenizer,
#             batch_size=batch_size, 
#             sequence_length=sequence_length,
#             rows=pages
#         )
#     )
    
#     # Reset gradients in the model to zero
#     model.zero_grad()
    
#     # Process each batch of data
#     for batch in batches:
#         # Move the batch to the specified device
#         inputs = batch.to(device)
#         # Pass the inputs through the model and calculate the loss
#         outputs = model(inputs, labels=inputs)
#         # Normalize the loss by the number of batches
#         outputs.loss /= len(batches)
#         # Backpropagate the loss to compute gradients
#         outputs.loss.backward()
#         # Exit the loop after processing the first batch for demonstration purposes
#         break
    
#     # Extract the top-k percent gradients from the model
#     gradient = topk_gradient(model, topk_percent)
    
#     # Return the top-k gradients
#     return gradient

    
# def check_equality(
#         proof_A: typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]],
#         proof_B: typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]],
#     ):
#     """
#     Checks if two gradient proofs are equal.

#     Args:
#         proof_A (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): The first gradient proof.
#         proof_B (Dict[str, Tuple[torch.Tensor, torch.Tensor]]): The second gradient proof.

#     Returns:
#         bool: True if the proofs are equal, False otherwise.
#     """
#     # Iterate through each item in the first proof
#     for key, (indices_A, values_A) in proof_A.items():
#         # Check if the key exists in the second proof
#         if key not in proof_B:
#             # If not, the proofs are not equal
#             return False
#         # Retrieve the corresponding item from the second proof
#         indices_B, values_B = proof_B[key]
        
#         # Convert data types for comparison
#         indices_A = indices_A.to(torch.int32)
#         values_A = values_A.to(torch.float32)
#         indices_B = indices_B.to(torch.int32)
#         values_B = values_B.to(torch.float32)
        
#         # Check if the indices and values are equal
#         if not (torch.equal(indices_A, indices_B) and torch.allclose(values_A, values_B)):
#             # If not, the proofs are not equal
#             return False
#     # If all checks pass, the proofs are equal
#     return True
