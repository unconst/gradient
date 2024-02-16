
import argparse
import bittensor as bt

parser = argparse.ArgumentParser(description="Validator config")
parser.add_argument("--device", type=str, default="cpu", help="Device to use for computations.")
parser.add_argument("--netuid", type=int, default=81, help="Netuid to mine on for Bittensor.")
bt.wallet.add_args(parser)
bt.logging.add_args(parser)
bt.subtensor.add_args(parser)
config = bt.config(parser)

bt.logging( config = config )
wallet = bt.wallet( config = config )
subtensor = bt.subtensor( config = config )
metagraph = subtensor.metagraph( config.netuid )

scores = { uid:0 for uid in range(metagraph.n) }
while True:
    model = get_model()
    for uid in metagraph.uids:
        grad_files = get_grad_files( uid, model )
        for full_grad in grad_files:
            if random.random() < verify_rate:
                real_gradient = get_full_gradient( full_grad )
                if not verify_grad( full_grad, model ):
                    scores[ uid ] = loss_alpha * scores[ uid ] - (1-loss_alpha) * 0
                else:
                    scores[ uid ] = win_alpha * scores[ uid ] - (1 - win_alpha) * 1  
        
        
    

    


