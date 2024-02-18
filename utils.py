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
import torch
import typing
import hashlib

def accumulate_gradient(model, grads):
    """
    Accumulates multiple gradient proofs onto a model's gradients.

    Args:
        model (torch.nn.Module): The model to update.
        grad (List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]): The grad to accumulate.

    """
    # Iterate through each parameter in the model
    for name, param in model.named_parameters():
        # Check if the parameter requires gradients and has not been processed yet
        if not param.requires_grad: continue
        if param.grad is None: param.grad = torch.zeros_like(param.data)
        
        # Initialize a tensor to accumulate gradients and a counter for averaging
        accumulated_grad = torch.zeros_like(param.grad.view(-1))
        grad_count = 0
        
        # Iterate through each proof
        for g in grads:
            # Check if the current parameter is present in the proof
            if name not in g: continue
            # Retrieve the indices and values from the proof
            indices, values = g[name]
            
            # Accumulate the values from the proof to the corresponding indices in the gradient tensor
            accumulated_grad.scatter_add_(0, indices.to(torch.long), values)
            grad_count += 1
        
        # Average the accumulated gradients if there are any gradients to average
        if grad_count > 0:
            averaged_grad = accumulated_grad / grad_count
        else:
            averaged_grad = accumulated_grad
        
        # Reshape the averaged gradient tensor back to its original shape and assign it to param.grad
        param.grad = averaged_grad.view_as(param.grad)

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

def create_model_hash(model: torch.nn.Module ):
    """
    Generates a SHA-256 hash of the model's state dictionary by iterating through the values of each item.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: The SHA-256 hash of the model's state dictionary.
    """
    # Extract the state dictionary from the model which contains all the parameters
    model_state_dict = model.state_dict()
    # Concatenate all the model state values into a single string
    concatenated_model_states = ''.join([ str(value.cpu().numpy().tobytes()) for value in model_state_dict.values()])
    # Encode the concatenated string into bytes
    concatenated_model_states_bytes = concatenated_model_states.encode()
    # Generate a SHA-256 hash from the concatenated bytes
    return hashlib.sha256(concatenated_model_states_bytes).hexdigest()