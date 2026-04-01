import torch
import torch.nn as nn

class CounterfactualLoss(nn.Module):
    def __init__(self, mask_ratio=0.2):
        super(CounterfactualLoss, self).__init__()
        self.mask_ratio = mask_ratio

    def forward(self, model, features, captions, pad_idx, attribution_maps, padding_mask=None):
        B, seq_len, num_pixels = attribution_maps.shape
        
        with torch.no_grad():
            orig_outputs, _ = model.decoder(features, captions, pad_idx)
            orig_probs = torch.softmax(orig_outputs, dim=-1)
            target_words = captions.unsqueeze(-1) 
            orig_target_probs = orig_probs.gather(2, target_words).squeeze(-1) 
            
        total_cf_loss = 0.0
        valid_tokens = 0
        

        for t in timesteps:
            if padding_mask is not None and padding_mask[:, t].all():
                continue # Skip if entirely padded
                
            attr_map_t = attribution_maps[:, t, :].detach() 
            
            k = max(1, int(num_pixels * self.mask_ratio))
            topk_vals, _ = torch.topk(attr_map_t, k, dim=-1)
            thresholds = topk_vals[:, -1].unsqueeze(-1)
            
            # Create mask and apply
            keep_mask = (attr_map_t < thresholds).unsqueeze(-1).float() 
            masked_features = features * keep_mask
            
            # Re-run decoder with masked features ONLY for this random valid step
            masked_outputs, _ = model.decoder(masked_features, captions, pad_idx)
            masked_probs = torch.softmax(masked_outputs, dim=-1)
            
            target_word_t = target_words[:, t, :] 
            masked_target_prob_t = masked_probs[:, t, :].gather(1, target_word_t).squeeze(-1) 
            orig_target_prob_t = orig_target_probs[:, t] 
            
            drop = orig_target_prob_t - masked_target_prob_t
            
            if padding_mask is not None:
                step_valid = ~padding_mask[:, t]
                drop = drop * step_valid
                valid_tokens += step_valid.sum()
                total_cf_loss = total_cf_loss - drop.sum()
            else:
                valid_tokens += B
                total_cf_loss = total_cf_loss - drop.sum()
                

