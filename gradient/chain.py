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
import bittensor as bt


def try_set_weights( 
        weights: torch.FloatTensor, 
        wallet: 'bt.wallet',
        subtensor: 'bt.subtensor', 
        netuid: int, 
        ttl: int 
    ):
    def _try_set_weights():
        try:
            weights.nan_to_num(0.0)
            subtensor.set_weights(
                netuid = netuid,
                wallet=self.wallet,
                uids=self.metagraph.uids,
                weights=self.weights,
                wait_for_inclusion=False,
                version_key=constants.weights_version_key,
            )
        except:
            bt.logging.warning("Failed to set weights. Trying again later.")

        ws, ui = self.weights.topk(len(self.weights))
        table = Table(title="All Weights")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

    try:
        bt.logging.debug(f"Setting weights.")
        await asyncio.wait_for(_try_set_weights(), ttl)
        bt.logging.debug(f"Finished setting weights.")
    except asyncio.TimeoutError:
        bt.logging.error(f"Failed to set weights after {ttl} seconds")

    async def try_sync_metagraph(self, ttl: int):
        def sync_metagraph(endpoint):
            metagraph = bt.subtensor(endpoint).metagraph(self.config.netuid)
            metagraph.save()

        process = multiprocessing.Process(
            target=sync_metagraph, args=(self.subtensor.chain_endpoint,)
        )
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f"Failed to sync metagraph after {ttl} seconds")
            return

        bt.logging.info("Synced metagraph")
        with self.metagraph_lock:
            self.metagraph.load()
            self.miner_iterator.set_miner_uids(self.metagraph.uids.tolist())
            self.model_tracker.on_hotkeys_updated(set(self.metagraph.hotkeys))
