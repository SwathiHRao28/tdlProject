import torch
import torch.nn as nn

class AlignmentLoss(nn.Module):
    def __init__(self):
        super(AlignmentLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, attention_maps, attribution_maps, padding_mask=None):
        """
        Calculates MSE between attention maps and attribution maps.
        
        Args:
            attention_maps (Tensor): (B, seq_len, num_pixels)
            attribution_maps (Tensor): (B, seq_len, num_pixels)
            padding_mask (Tensor): (B, seq_len) Boolean mask where True means padded token
            
        Returns:
            Scalar loss
        """
        # Normalize attention maps per token to [0, 1] range to match attribution
        min_attn = attention_maps.min(dim=-1, keepdim=True)[0]
        max_attn = attention_maps.max(dim=-1, keepdim=True)[0]
        val_range = max_attn - min_attn
        val_range[val_range == 0] = 1e-8
        
        norm_attention_maps = (attention_maps - min_attn) / val_range
        
        if padding_mask is not None:
            # Mask out padded tokens
            # padding_mask shape is (B, seq_len)
            mask = ~padding_mask
            mask = mask.unsqueeze(-1) # (B, seq_len, 1)
            
            norm_attention_maps = norm_attention_maps * mask
            attribution_maps = attribution_maps * mask
            
            # Compute MSE only over valid tokens
            loss = nn.functional.mse_loss(
                norm_attention_maps, 
                attribution_maps, 
                reduction='sum'
            ) / (mask.sum() * attention_maps.size(-1) + 1e-8)
            
            return loss
            
        return self.mse(norm_attention_maps, attribution_maps)
