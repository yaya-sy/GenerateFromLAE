import sys
sys.path.append('.')
from scripts.model.decoder import Decoder
from scripts.model.encoder import Encoder
from scripts.configs.config import Config
from typing import Optional

import yaml
import torch
from torch import nn
from torch import Tensor

class Seq2Seq(nn.Module):
    """ TODO """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad_idx = config.pad_idx
        self.input_w_embeddings = nn.Embedding(num_embeddings=config.vocab_size,
                                               embedding_dim=config.embedding_dims,
                                               padding_idx=config.pad_idx)
        self.output_w_embeddings = nn.Embedding(num_embeddings=config.vocab_size,
                                                embedding_dim=config.embedding_dims,
                                                padding_idx=config.pad_idx)
        self.pos_embeddings = None
        if self.config.src_pos_embeddings:
            self.pos_embeddings = nn.Embedding(num_embeddings=config.max_length,
                                                    embedding_dim=config.embedding_dims,
                                                    padding_idx=config.pad_idx)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.linear = nn.Linear(config.embedding_dims, config.vocab_size)
    
    def forward(self, src: Tensor, tgt: Tensor):
        """ TODO """
        _, src_s = src.shape
        _, tgt_s = tgt.shape
        device = src.device
        
        # embeddings
        src_embeddings = self.input_w_embeddings(src)
        if self.pos_embeddings is not None:
            src_positions = torch.arange(0, src_s, dtype=torch.long, device=device).unsqueeze(0)
            src_p_embeddings = self.pos_embeddings(src_positions)
            src_embeddings += src_p_embeddings

        tgt_embeddings = self.output_w_embeddings(tgt)
        tgt_positions = torch.arange(0, tgt_s, dtype=torch.long, device=device).unsqueeze(0)
        tgt_p_embeddings = self.pos_embeddings(tgt_positions)
        tgt_embeddings += tgt_p_embeddings

        # masking
        src_decoder_mask = torch.zeros(tgt_s, src_s).bool().to(device)
        src_encoder_mask = torch.zeros(src_s, src_s).bool().to(device)
        src_pad_mask = (src == self.pad_idx)[-1, ...].to(device)
        src_decoder_mask[:, src_pad_mask] = True
        src_encoder_mask[:, src_pad_mask] = True

        tgt_mask = torch.ones(tgt_s, tgt_s).triu_(1).bool().to(device)
        tgt_pad_mask = (tgt == self.pad_idx)[-1, ...].to(device)
        tgt_mask[:, tgt_pad_mask] = True

        # encoder-decoder
        encoded_src = self.encoder(src_embeddings, src_encoder_mask)
        decoded_tgt = self.decoder(encoded_src, tgt_embeddings, src_decoder_mask, tgt_mask)

        return self.linear(decoded_tgt)

# with open("configs/mini.yml", "r") as config_file:
#     yaml_config = yaml.safe_load(config_file)
#     config = Config(**yaml_config)

# model = Seq2Seq(config)

# src = torch.tensor([[i for i in range(5)] for _ in range(10)])
# tgt = torch.tensor([[i for i in range(7)] for _ in range(10)])

# out = model(src, tgt)
# print(out)
# print(out.shape)