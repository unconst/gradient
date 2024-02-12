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
import pickle
import hashlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data import SubsetFalconLoader
from protocol import Seal

def get_model_and_tokenizer():
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

def topk_gradient(model: torch.nn.Module, topk_percent: float) -> typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]]:
    gradient_data = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            total_elements = parameter.grad.numel()
            topk_elements = int(total_elements * topk_percent)
            values, indices = torch.topk(parameter.grad.abs().flatten(), topk_elements)
            # Cast indices to int32 and values to float32
            indices = indices.to(torch.int32)
            values = values.to(torch.float32)
            gradient_data[name] = (indices, values)
    return gradient_data

def create_proof( 
        model: torch.nn.Module,
        tokenizer: 'tokenizer',
        pages: typing.List[int], 
        batch_size: int, 
        sequence_length: int,
        device: str = 'cpu',
        topk_percent: float = 0.01
    ) -> typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]]:
        model.to( device )
        batches = list(
            SubsetFalconLoader(
                tokenizer = tokenizer,
                batch_size = batch_size, 
                sequence_length = sequence_length,
                rows = pages
            )
        )
        model.zero_grad()
        for batch in batches:
            inputs = batch.to(device)
            outputs = model( inputs, labels=inputs )
            outputs.loss /= len(batches)
            outputs.loss.backward()
            break
        
        gradient_data = topk_gradient(model, topk_percent)
        return gradient_data
    
def check_equality(
        proof_A: typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]],
        proof_B: typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]],
    ):
    for key, (indices_A, values_A) in proof_A.items():
        if key not in proof_B:
            return False
        indices_B, values_B = proof_B[key]
        # Ensure tensors are of the same type before comparison
        indices_A = indices_A.to(torch.int32)
        values_A = values_A.to(torch.float32)
        indices_B = indices_B.to(torch.int32)
        values_B = values_B.to(torch.float32)
        if not (torch.equal(indices_A, indices_B) and torch.allclose(values_A, values_B)):
            return False
    return True


def accumulate_proofs(model, proofs):
    """
    Accumulate multiple proofs by scatter adding them onto the grad values of the model.
    
    Args:
    - model (torch.nn.Module): The model whose gradients will be updated.
    - proofs (List[Dict[str, Tuple[torch.LongTensor, torch.FloatTensor]]]): A list of proofs, 
      where each proof is a dictionary mapping parameter names to tuples of indices and values.
    """    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        # Initialize a tensor of zeros with the same shape as the parameter's gradient
        if param.grad is None: param.grad = torch.zeros_like(param.data)
        # Flatten the gradient to apply scatter add
        grad_flat = param.grad.view(-1)
        # Iterate through each proof and scatter add the values onto the flattened gradient
        for proof in proofs:
            if name not in proof: continue
            indices, values = proof[name]
            # Since the gradient is flattened, indices do not need to be unraveled
            grad_flat.scatter_add_(0, indices.to(torch.long), values)
        # Unflatten the gradient back to its original shape
        param.grad = grad_flat.view_as(param.grad)
        
        
def create_seal(model, proof, pages, batch_size, sequence_length, topk_percent):
    """
    Compute a hash of the proof, the model, and the parameters used to generate the gradient proof.
    
    Args:
    - model (torch.nn.Module): The model used to generate the proofs.
    - proofs (Dict[str, Tuple[torch.LongTensor, torch.FloatTensor]]): proof
    - pages (List[int]): The pages used to generate the proofs.
    - batch_size (int): The batch size used.
    - sequence_length (int): The sequence length used.
    - topk_percent (float): The topk percent used.
    
    Returns:
    - Seal: The computed seal containing the hashes and parameters.
    """
    # Serialize model's state dict and compute its hash
    model_state_dict = model.state_dict()
    model_state_bytes = pickle.dumps(model_state_dict)
    model_hash = hashlib.sha256(model_state_bytes).hexdigest()
    
    # Convert all other inputs to string and concatenate
    concatenated_inputs = ''.join([str(item) for item in proof.values()]) + \
                          ''.join([str(page) for page in pages]) + \
                          str(batch_size) + str(sequence_length) + str(topk_percent)
    # Encode the concatenated string
    encoded_inputs = concatenated_inputs.encode()
    # Compute the hash for the gradient proof
    gradient_hash = hashlib.sha256(encoded_inputs).hexdigest()
    
    return Seal(
        pages = pages,
        model_hash = model_hash,
        gradient_hash = gradient_hash,
        batch_size = batch_size,
        sequence_length = sequence_length,
        topk_percent = topk_percent
    )
