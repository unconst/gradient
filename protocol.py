
import typing
import bittensor as bt

class Verify( bt.Synapse ):
    pages: typing.List[int]
    proof_hash: str
    signature: bytes
    sequence_length: int
    batch_size: int
    topk_percent: float