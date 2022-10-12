from typing import Optional

from blocks import DecoderBlock
from torch import nn
from torch import Tensor

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
    
    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Optional[Tensor],
                tgt_mask: Optional[Tensor]) -> Tensor:
        """
        Forward an embedded sequence.
        
        Parameters
        ----------
        - src: Tensor
            The transformed (contextualized) source sequences.
            Must be of shape [b, s, c] where 'b' is the batch size,\
            's' the sequences lengths and 'c' the hidden vectors dimension.
        - tgt: Tensor
            The embedded (non contextualized) target sequences.
            Must be of shape [b, s, e] where 'b' is the batch size,\
            's' the sequences lengths and 'e' the embeddings dimension.
        - src_mask: Optional
            If given, will be used to ignore values to not be attended in the source sequences.
            Must be of shape [tgt_s, src_s] where 'tgt_s' is the length of the target sequences,\
            and 'src_s' is length of source sequences.
        - tgt_mask: Optional
            If given, will be used to ignore values to not be attended in the target sequences.
            Must be of shape [tgt_s, tgt_s] where 'tgt_s' is the length of the target sequences.
        
        Returns
        -------
        - Tensor:
            The target hidden word vectors that have attended to the source sequence.
            Shape of [b, tgt_s, c] where 'b' is the batch size, 'tgt_s' the length of\
            target sequence and 'c' is the size of the hidden vector.
        """
        for layer in self.layers:
            tgt = layer(src, tgt, src_mask, tgt_mask)
        return tgt