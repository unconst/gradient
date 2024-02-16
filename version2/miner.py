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

import random
import argparse
import bittensor as bt

parser = argparse.ArgumentParser(description="Validator config")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
parser.add_argument("--netuid", type=int, default=81, help="Netuid to mine on for Bittensor.")
parser.add_argument("--bucket", type=str, default='my_bucket_name', help="Name of my bucket.")
bt.wallet.add_args(parser)
bt.logging.add_args(parser)
bt.subtensor.add_args(parser)
config = bt.config(parser)

bt.logging( config = config )
wallet = bt.wallet( config = config )
subtensor = bt.subtensor( config = config )
subtensor.commit( wallet, config.bucket )

def get_model():
    
while True:
    model = get_model()
    model_id = get_model_id()
    while model_id == get_model_id():
        model.zero_grad()
        batches = Falcon( pages = random.randint( 900000000 ) )
        for inputs in batches:
            output = model( inputs, labels = inputs )
            output.loss.backward()
        store_gradients( config.bucket, model )
    


