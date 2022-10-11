"""Implementation of one transformer layer."""
# standard python imports
import sys
sys.path.append('.')
from typing import Optional
from abc import ABC, abstractmethod
from scripts.configs.config import Config

# non standard python libraries imports
import torch
from torch import nn
from torch import Tensor

import yaml

class MultiHeadAttention(nn.Module):
    """
    This class implements a multihead attention.

    Parameters
    ----------
    - config: Config
        An instance of the class Config containing all the hyperparameters
        and options.
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.K = nn.Linear(in_features=config.embedding_dims, out_features=config.embedding_dims)
        self.Q = nn.Linear(in_features=config.embedding_dims, out_features=config.embedding_dims)
        self.V = nn.Linear(in_features=config.embedding_dims, out_features=config.embedding_dims)
        self.heads = config.heads
        self.softmax = nn.Softmax(-1)
        self.dropout_mha = nn.Dropout(p=config.dropout)
    
    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Optional[Tensor]=None
                ) -> Tensor:
        """
        Compute attention mechanism of an embedded sequence of tokens.

        Parameters
        ----------
        - q: Tensor
            The queries as a three dimension tensor of embedded sequences.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the lengths of the sequences and 'e' the embedding dimension.
            
        - k: Tensor
            The keys as a three dimension tensor of embedded sequences.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the lengths of the sequences and 'e' the embedding dimension.

        - v: Tensor
            The values as a three dimension tensor of embedded sequences.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the length of the sequence and 'e' the embedding dimension.
        - mask: Tensor
            The keys vector to not attend. Optional. If given, must be\
            of shape [s_q, s_k] where 's_q' is the sequence length of the queries\
            and 's_k' is the sequence length of the keys.
        
        Returns
        -------
        - Tensor:
            Tensor of shape [b, s_q, c] where 'b' is the batch size,\
            's_q' is the length of the query sequence and 'c' the reduction of\
            the concatenation of the output of each attention model.
            'c' is reduced since the keys, queries and values projections\
            of the tokens of the sequence are divided by the number of heads.

            The returned vector represents for each tokens the concatenation\
            of its attention vectors coming from all the heads.
        """
  
        b, s_q, _ = q.shape # [batch_size, seq_length, embedd_dims]
        _, s_k, _ = k.shape 
        _, s_v, _ = v.shape

        # linear projections of the keys, queries and values.
        # and split them into the number of heads, before to compute the attention
        # so we have multiple attention models (heads) learned independantly.
        K = self.K(k).view(b, self.heads, s_k, -1).transpose(2, 3)
        Q = self.Q(q).view(b, self.heads, s_q, -1)
        V = self.V(v).view(b, self.heads, s_v, -1)

        # dot-product and use the scaling factor from 'Attention is all you need" paper
        # (https://arxiv.org/pdf/1706.03762.pdf)
        QK = (Q @ K) / 1 / torch.sqrt(torch.tensor(K.shape[-1])) # shape=[b, h, s_q, s_k]
        print(QK.shape)
        if mask is not None:
            QK += mask[:s_q, :s_k]
        attention = self.softmax(QK) # shape=[b, h, s_q, s_k])
        # for each word, concatenate the attention vectors comming from all the heads.
        out = (attention @ V).view(b, s_q, -1) # [b, s, embedd_dims]
        return self.dropout_mha(out)

class Block(ABC, nn.Module):
    """
    This class implements one tranfromer block.

    Parameters
    ----------
    - config: Config
        An instance of the class Config containing all the hyperparameters
        and options.
    """
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=config.embedding_dims, out_features=config.ff_size),
            nn.GELU(),
            nn.Linear(in_features=config.ff_size, out_features=config.embedding_dims)
        )

        self.dropout_ff = nn.Dropout(p=config.dropout)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.embedding_dims)
    
    @abstractmethod
    def forward(self,
                src: Tensor,
                tgt: Tensor,
                mask: Optional[Tensor]=None
                ) -> Tensor:
        """
        Forward an embedded tensor in the transformer layer.

        Parameters
        ----------
        - src: Tensor
            The three dimension tensor of the embedded sequence.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the length of the source sequence and 'e' the embedding dimension.
        - tgt: Tensor
            The three dimension tensor of the embedded sequence.
            The shape must be [b, s, e] where 'b' is the batch size,\
            's' is the length of the target sequence and 'e' the embedding dimension.

        - mask: Tensor or None.
            Mask the pad indexes. If given, must be a shape of [s_tgt, s_src] where\
            s_tgt is the length of the target sequence and s_src is the length of the source sequence. 

        
        Returns
        -------
        - Tensor:
            Tensor of shape [b, s, c] where 'b' is the batch size,\
            's' is the length of the sequence and 'c' the contextual\
            embedding dimension.
            This represents the contextual vector for each vector in the\
            the sequence. So each vector in the sequence contains\
            the informations about other tokens in the sequence.
        """
        pass

class EncoderBlock(Block):
    def __init__(self, config):
        super().__init__(config)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                mask: Optional[Tensor]) -> Tensor:

        self_attended = self.mha(src, tgt, tgt, mask)
        self_add_norm = self.layer_norm1(self_attended + tgt)
        ffw = self.dropout_ff(self.mlp(self_add_norm))
        return self.layer_norm2(ffw + self_add_norm)

class DecoderBlock(Block):
    def __init__(self, config):
        super().__init__(config)
        self.additional_mha = MultiHeadAttention(config)
        self.additional_layer_norm = nn.LayerNorm(normalized_shape=config.embedding_dims)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                pad_mask: Optional[Tensor]=None,
                tgt_mask_futur: Optional[Tensor]=None) -> Tensor:

        tgt_attended = self.additional_mha(tgt, tgt, tgt, tgt_mask_futur)
        tgt_add_norm = self.additional_layer_norm(tgt_attended + tgt)
        cross_attended = self.mha(tgt, src, src, pad_mask)
        cross_add_norm = self.layer_norm1(cross_attended + tgt_add_norm)
        ffw = self.dropout_ff(self.mlp(cross_add_norm))
        return self.layer_norm2(ffw + cross_add_norm)

with open("configs/mini.yml", "r") as config_file:
    yaml_config = yaml.safe_load(config_file)
    config = Config(**yaml_config)
decoder = DecoderBlock(config)
# TODO: Try masking
pad_mask = None
src = torch.rand(10, 5, 256)
tgt = torch.rand(10, 7, 256)
out = decoder(src, tgt)
print(out)
print(out.shape)