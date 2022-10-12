from typing import Optional

from blocks import EncoderBlock
from torch import nn
from torch import Tensor

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.layers)])
    
    def forward(self,
                src: Tensor,
                mask: Optional[Tensor]) -> Tensor:
        """
        Forward an embedded sequence.
        
        Parameters
        ----------
        - src: Tensor
            The embedded sequence to encode.
            Must be of shape [b, s, e] where 'b' is the batch size,\
            's' the sequence length and 'e' the embedding dimensions.
        - mask: Optional
            If given, will be used to ignore values to not be attended.
            Must be of shape [s, s] where 's' is the length of the source\
            tensor. 
        
        Returns
        -------
        - Tensor:
            The hidden tokens vectors of the source sequence, where each vector\
            is contextualized thanks to the self attention.
            Shape of [b, s, c] where 'b' is the batch size, 's' is the source sequences lengths\
            and 'c' is the hidden vectors dimension.
        """
        for layer in self.layers:
            src = layer(src, mask)
        return src