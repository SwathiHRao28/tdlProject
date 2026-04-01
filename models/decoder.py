import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x has shape (batch_size, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask, need_weights=False)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention (we NEED the attention weights here for Alignment Loss)
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory,
                                                 key_padding_mask=memory_key_padding_mask,
                                                 need_weights=True,
                                                 average_attn_weights=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_weights

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, nhead, max_length=50):
        super(CaptionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Fix PositionalEncoding size mismatch error by defaulting to a large max_len table
        self.pos_encoder = PositionalEncoding(embed_size, max_len=max(max_length, 5000))
        
        # Linear layer to project visual features to text hidden dimension if they differ
        # Assuming visual features will be projected to `embed_size` outside, or just setting hidden_size=embed_size
        assert embed_size == hidden_size, "For simplicity, embed_size must equal hidden_size"
        
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, features, captions, pad_idx=None):
        """
        features: (B, num_pixels, hidden_size)
        captions: (B, seq_len)
        """
        seq_len = captions.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len, captions.device)
        
        tgt_key_padding_mask = (captions == pad_idx) if pad_idx is not None else None
        
        x = self.embed(captions) # (B, seq_len, embed_size)
        x = self.pos_encoder(x)
        
        # Pass through individual layers
        # Store cross-attention weights of the last layer
        last_attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(
                tgt=x,
                memory=features,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            last_attn_weights = attn_weights # shape: (B, seq_len, num_pixels)
            
        outputs = self.fc_out(x)
        return outputs, last_attn_weights
